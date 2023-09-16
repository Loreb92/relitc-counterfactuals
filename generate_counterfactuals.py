import os
import argparse
import pickle
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import torch

from src.utils import print_msg
from extract_feature_importances import load_blackbox_and_tokenizer
from src.Counterfactual_pipeline.BlackBox.blackbox import BlackBoxTransformersForSequenceClassification
from src.Counterfactual_pipeline.Editor.search_counterfactuals import run_binary_search
from src.Counterfactual_pipeline.Editor.CMLM.editor import EditorTransformersForCMLM
from src.Counterfactual_pipeline.Editor.CMLM.masker_dataset import MaskerDataset
from src.Counterfactual_pipeline.Editor.counterfactual_evaluation import MaskedLanguageModelScorer, SemanticSimilarityScorer, MiceMinimalityScorer
from src.Counterfactual_pipeline.Editor.edit_scorer_from_mice import EditEvaluator as EditEvaluatorMice


def add_generation_config_args(parser):
    
    group = parser.add_argument_group('generation', 'generation configurations')
    group.add_argument('--editor-model-path', type=str, help="Location of the editor model")
    group.add_argument('--sampling', type=bool, help="Whether to use sampling heuristic for generation")
    group.add_argument('--top-p', type=float, help="Top-p parameter for nucleous sampling")
    group.add_argument('--top-k', type=int, help="Top-k parameter for decoding")
    group.add_argument('--next-token-strategy', type=str, help="Strategy to choose the order of tokens to infill", choices=['ordered', 'confidence'])
    group.add_argument('--direction', type=str, help="Strategy to choose the direction of tokens to infill, given the next-token-strategy", choices=['left_to_right', 'right_to_left', 'highest_first', 'lowest_first'])
    group.add_argument('--max-search-levels', type=int, help="Maximum number of binary search steps")
    group.add_argument('--n-edited-texts', type=int, help="Maximum number of candidate counterfactuals generated per step")
    
    return parser


def add_evaluation_config_args(parser):
    
    group = parser.add_argument_group('evaluation', 'evaluation configurations')
    group.add_argument('--fluency-model-path', type=str, help="Location of the fluency model")
    group.add_argument('--sem-similarity-model-path', type=str, help="Location of the semantic similarity model")
    
    return parser
    

def get_class_to_counterfactual_class_name(dataset):
    """
    Mapping from text's label to counterfactual label name.
    """
    
    if dataset == 'Yelp':
        class2counterf_class_name = {
            0:'positive',
            1:'negative'
        }        
    elif dataset == 'OLID':
        class2counterf_class_name = {
            0:'offensive',
            1:'not offensive'
        }
    elif dataset == 'yelp_sentence':
        class2counterf_class_name = {
            0:'positive',
            1:'negative'
        } 
    elif dataset == 'call_me':
        class2counterf_class_name = {
            0:'sexist', 
            1:'non-sexist'
        }
    else:
        raise NotImplementedError(f"Can not read the dataset {dataset}: functions not implemented.")
        
    return class2counterf_class_name


def get_label_to_counterfactual_label(dataset):
    
    if dataset == 'Yelp':
        label_name2counterf_label_name = {
            'negative':'positive',
            'positive':'negative'
        }
    elif dataset == 'OLID':
        label_name2counterf_label_name = {
            'not offensive':'offensive',
            'offensive':'not offensive'
        }
    elif dataset == 'yelp_sentence':
        label_name2counterf_label_name = {
            'negative':'positive',
            'positive':'negative'
        }
    elif dataset == 'call_me':
        label_name2counterf_label_name = {
            'non-sexist':'sexist',
            'sexist':'non-sexist'
        }
    else:
        raise NotImplementedError(f"Can not read the dataset {dataset}: functions not implemented.")
        
    return label_name2counterf_label_name

def load_dataset(dataset, data_path, explain_wrt, tokenizer, id2label, frac_mask_words, seed):
    
    data_fold = os.path.join(data_path, dataset)

    # load data
    test_df = pd.read_json(os.path.join(data_fold, f"test_with_token_importances_wrt_{explain_wrt}.json"),
                            orient='records', lines=True)
    
    # make dataset
    test_dataset = MaskerDataset(test_df, tokenizer, id2label, use_ctrl_code=True, frac_mask_words=frac_mask_words, seed=seed)
    
    return test_dataset


def load_editor_and_blackbox(args):
    
    # load editor
    print_msg(f"Loading editor from {args.editor_model_path}...")
    editor = EditorTransformersForCMLM(args.editor_model_path,
                                        sampling=args.sampling,
                                        top_p=args.top_p,
                                        top_k=args.top_k, 
                                       next_token_strategy=args.next_token_strategy,
                                       direction=args.direction, 
                                       seed=args.seed)
    editor.model_to_device()
    
    # load blackbox
    print_msg(f"Loading blackbox of datset {args.dataset} from {args.blackbox_model_path}...")
    blackbox, class2label = load_blackbox_and_tokenizer(args.dataset, args.blackbox_model_path)
    blackbox.model_to_device()
    
    return editor, blackbox, class2label


def eval_counterfactuals(results, eval_classes_dict):
    """
    Evaluate counterfactuals.
    
    Parameters:
    results : dict
    eval_classes_dict : dict, contains the metric classes as values and the name of the metric as key
    
    Returns:
    counterfactuals : list, contains all successful counterfactuals with the metrics
    
    """
    
    original_text = results['original_text']
    
    # keep only successful counterfactuals
    counterfactuals = [candidate for candidate in results['results'] if candidate['success']]
    counterfactual_texts = [counterfactual['edited_output_text'] for counterfactual in counterfactuals]
    
    # compute metrics for all counterfactuals and add it to counterfactuals
    for metric, eval_class in eval_classes_dict.items():
        
        scores = eval_class.score_texts(original_text, counterfactual_texts)
        counterfactuals = [{**counterfactual, metric:score} for counterfactual, score in zip(counterfactuals, scores)]
        
    return counterfactuals   


def load_eval_classes(fluency_model_name, sem_similarity_model_name):
    
    print_msg(f"Loading fluency scorer model from {fluency_model_name}...")
    fluency_scorer = MaskedLanguageModelScorer(fluency_model_name, batch_size=64)
    fluency_scorer.model_to_device()
    sem_similarity_scorer = SemanticSimilarityScorer(sem_similarity_model_name, batch_size=32)
    print_msg(f"Loading semantic similarity scorer model from {sem_similarity_model_name}...")
    minimality_scorer = MiceMinimalityScorer()
    
    # load also MiCe Fluency scorer
    print_msg(f"Loading MiCe fluency scorer...")
    fluency_mice_scorer = EditEvaluatorMice()
    
    return {'fluency':fluency_scorer, 'sem_similarity':sem_similarity_scorer, 'minimality':minimality_scorer, 'fluency_mice':fluency_mice_scorer }
    

def main(args):
    
    # load editor and black box
    Editor, Blackbox, _ = load_editor_and_blackbox(args)
    label_name2counterf_label_name = get_label_to_counterfactual_label(args.dataset)
    class_to_counterfactual_class_name = get_class_to_counterfactual_class_name(args.dataset)
    
    # load eval classes
    eval_classes_dict = load_eval_classes(args.fluency_model_path, args.sem_similarity_model_path)
    
    # load data
    # load_dataset takes already the mapping from instance's class to counterfactual class
    print_msg("Loading data...")
    frac_mask_words = None
    test_dataset = load_dataset(args.dataset, args.data_path, args.explain_wrt, Editor.tokenizer, class_to_counterfactual_class_name, frac_mask_words, seed=None)
    
    # check results file
    results_file = os.path.join(args.results_folder, args.dataset)
    if not os.path.exists(results_file):
        os.mkdir(results_file)
        
    if not args.baseline_editor:
        results_file = os.path.join(results_file, f'counterfactuals_explain_wrt-{args.explain_wrt}_next_token_strategy-{args.next_token_strategy}_direction-{args.direction}.json')
    else:
        results_file = os.path.join(results_file, f'counterfactuals_baseline_explain_wrt-{args.explain_wrt}_next_token_strategy-{args.next_token_strategy}_direction-{args.direction}.json')
    
    print_msg(f"Writing results on file: {results_file}")
    if os.path.exists(results_file):
        print_msg("Results file already exists, do not overwrite.")
        return 1
            
    # loop across data instances to generate and evaluate counterfactuals
    for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
    
        item = test_dataset.df.loc[i]
        info_item = {
            'id':item.id,
            'text':item.text,
            'label':int(item.label),
            'original_label':int(item.original_label),
            'pred_label':int(item.pred_label),
            'pred_proba':float(item.pred_proba)
        }
        
        
        try:
            # search counterfactuals
            t0 = time.time()
            results_search_hist = run_binary_search(Editor=Editor, 
                                                   Blackbox=Blackbox,
                                                   dataset=test_dataset, 
                                                   idx=i, 
                                                   min_mask_frac=args.min_mask_frac, 
                                                   max_mask_frac=args.max_mask_frac,
                                                   max_search_levels=args.max_search_levels, 
                                                   n_edited_texts=args.n_edited_texts, 
                                                   results=None)
            t_search = time.time() - t0
            
            # get the successful runs with minimum mask fraction
            results_success = [result for result in results_search_hist if result['success']]
            is_any_success = bool(len(results_success) > 0)
            if is_any_success:
                t0 = time.time()
                results_best = min(results_success, key=lambda item: item['mask_frac'])
                counterfactuals = eval_counterfactuals(results_best, eval_classes_dict)
                t_eval = time.time() - t0
                mask_frac = results_best['mask_frac']
            else:
                # no successful counterfactuals
                results_best = None
                counterfactuals = None
                mask_frac = None
                t_eval = None

            # write results
            line = {
                **info_item,
                'mask_frac':mask_frac,
                'success':is_any_success,
                'counterfactuals':counterfactuals,
                'results_best':results_best,
                'results_search_hist':results_search_hist,
                'time_generation':t_search,
                'time_evaluation':t_eval,
                'error':None
            }
            
            
        except KeyboardInterrupt:
            # keyboart interrupt command
            # write results
            line = {
                **info_item,
                'mask_frac':None,
                'success':None,
                'counterfactuals':None,
                'results_best':None,
                'results_search_hist':None,
                'time_generation':None,
                'time_evaluation':None,
                'error':'KeyboardInterrupt'
            }
            break
            
        except Exception as err:
            # error occurred, save the error
            line = {
                **info_item,
                'mask_frac':None,
                'success':None,
                'counterfactuals':None,
                'results_best':None,
                'results_search_hist':None,
                'time_generation':None,
                'time_evaluation':None,
                'error':err.args[0]
            }
            print_msg(err.args[0])
        
                               
        # store results
        with open(results_file, 'a') as ww:
            ww.write(json.dumps(line)+"\n")
            
    print_msg("Finished!")
    


if __name__ == '__main__':
    
    # parse arguments
    parser = argparse.ArgumentParser()    
    parser.add_argument('--data-path', type=str, help='The path of the folder containing the processed dataset.')
    parser.add_argument('--blackbox-model-path', type=str, help='The path of the folder containing the blackbox.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset to processs.')
    parser.add_argument('--explain-wrt', type=str, help='Get importances with respect to the original label or the predicted label.', choices=['original', 'predicted'])
    parser.add_argument('--seed', type=int, help='Random number generators seed', default=42)
    parser.add_argument('--min-mask-frac', type=float, help='Minimum token mask fraction', default=0.0)
    parser.add_argument('--max-mask-frac', type=float, help='Maximum token mask fraction', default=0.50)
    parser.add_argument('--results-folder', type=str, help='The folder where results are stored.')
    parser.add_argument('--baseline-editor', type=str, default="False", help='Whether to use non-finetuned editor.')
    
    
    # TODO: do we need to return? If yes, change also in finetune_conditional_masked_language_model.py
    add_generation_config_args(parser)
    add_evaluation_config_args(parser)
    args = parser.parse_args()
    
    args.baseline_editor = True if args.baseline_editor=='True' else False
    
    # set device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set model path
    if not args.baseline_editor:
        args.editor_model_path = os.path.join(args.editor_model_path, args.dataset, f"finetuned_CMLM_maks_frac-0.2_0.55-explain_wrt-{args.explain_wrt}")
    else:
        # use non fine tuned bert
        args.editor_model_path = os.path.join("/data/lbetti/models/bert-base-uncased")
    
    print_msg(args, with_time=False)
    main(args)