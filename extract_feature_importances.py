import argparse
import numpy as np
import os
import json
from tqdm import tqdm

import torch

from src.utils import print_msg
from src.feature_importance_scorer import FetureImportanceScorer
from src.Counterfactual_pipeline.BlackBox.blackbox import BlackBoxTransformersForSequenceClassification


def write_line(line, file):
    with open(file, 'a') as ww:
        ww.write(line+"\n")


def truncate_string(text, tokenizer, max_tokens=512):
    """
    Truncates the string to the desired number of tokens.
    """
    
    # add 2 to consider CLS and SEP tokens
    n_tokens = len(tokenizer.tokenize(text)) + 2
    n_chars = len(text)

    # if too long, binary search for max number of characters
    lb = 0
    ub = n_chars
    n_steps = 0
    while n_tokens>max_tokens or n_steps<=50:
        
        n_chars_cut = int(np.ceil((ub + lb)/2))

        text_ = text[:n_chars_cut]
        n_tokens = len(tokenizer.tokenize(text_)) + 2

        if n_tokens==max_tokens:
            text = text_
            break
        elif n_tokens<max_tokens:
            # diminish n_chars_cut
            lb = n_chars_cut
        else:
            # augment n_chars_cut
            ub = n_chars_cut
            
        n_steps += 1
                
                
    return text
        
           
def spans(text, fragments):
    """ Given a string and a list of tokens, returns the spans of each token in the text.
    """
    # https://stackoverflow.com/questions/43773962/how-can-i-find-the-position-of-the-list-of-substrings-from-the-string
    result = []
    point = 0  # Where we're in the text.
    for fragment in fragments:
        if fragment is None:
            result.append(None)
            continue
        try:
            found_start = text.index(fragment, point)
        except:
            print_msg(fragment)
            assert False
        found_end = found_start + len(fragment)
        result.append((found_start, found_end))
        point = found_end
    return result


# remove punctuation from things with feature importance
def align_feature_importances_with_offsets(text, tokens, token_importances):
    """
    Get the spans of tokens in text
    """

    # get the indices of the letters and numbers, then keep these tokens and corresponding importances
    is_alpha_idx = set([n for n, w in enumerate(tokens) if (w.isalpha() or w.isdigit()) and w!='Ċ'])

    tokens_classifier_alpha = [tok for n, tok in enumerate(tokens) if n in is_alpha_idx]
    tokens_classifier_alpha_importances = [score for n, score in enumerate(token_importances) 
                                               if n in is_alpha_idx]


    # get alignment
    tokens_classifier_alpha_spans = spans(text, tokens_classifier_alpha)
    
    return tokens_classifier_alpha, tokens_classifier_alpha_importances, tokens_classifier_alpha_spans

  
def read_data(dataset, data_path):
    """
    Read the dataset as a csv. This runs also basic text cleaning.
    
    Parameters:
    dataset : str, the name of the dataset.
    data_path : str, the path where the dataset is located.
    
    Returns:
    train_df, valid_df, test_df : pandas DataFrame, each has three columns ['id', 'text', 'label']
    """
    
    data_fold = os.path.join(data_path, dataset)
    
    if dataset == 'Yelp':
        from src.text_preprocessing.yelp import make_dataset

    elif dataset == 'OLID':
        from src.text_preprocessing.olid import make_dataset
        
    elif dataset == 'yelp_sentence':
        from src.text_preprocessing.yelp_sentence import make_dataset
        
    elif dataset == 'call_me':
        from src.text_preprocessing.call_me import make_dataset
                
    else:
        raise NotImplementedError(f"Can not read the dataset {dataset}: functions not implemented.")
        
    train_df, valid_df, test_df = make_dataset(data_fold)
        
    return train_df, valid_df, test_df


def load_blackbox_and_tokenizer(dataset, blackbox_path):
    
    if dataset == 'Yelp':
        model_fold = "textattack/bert-base-uncased-yelp-polarity"
        class2label = {0 : 'negative', 1 : 'positive'}
        blackbox = BlackBoxTransformersForSequenceClassification(model_fold, no_cuda=False, label_names=class2label)
        
    elif dataset == 'OLID':
        model_fold = "cardiffnlp/twitter-roberta-base-offensive"
        class2label = {0 : 'not offensive', 1 : 'offensive'}
        blackbox = BlackBoxTransformersForSequenceClassification(model_fold, no_cuda=False, label_names=class2label)
        
    elif dataset == 'yelp_sentence':
        model_fold = os.path.join(blackbox_path, "yelp_sentence_classifier")
        class2label = {0 : 'negative', 1 : 'positive'}
        blackbox = BlackBoxTransformersForSequenceClassification(model_fold, no_cuda=False, label_names=class2label)
        
    elif dataset == 'call_me':
        model_fold = os.path.join(blackbox_path, "call_me_classifier")
        class2label = {0 : 'non-sexist', 1 : 'sexist'}
        blackbox = BlackBoxTransformersForSequenceClassification(model_fold, no_cuda=False, label_names=class2label)
        
    else:
        raise NotImplementedError(f"Can not read the dataset {dataset}: functions not implemented.")
        
    return blackbox, class2label
        
    
    
def main(args):
    
    # read the dataset
    train_df, valid_df, test_df = read_data(args.dataset, args.data_path)
    
    # load blackbox
    # the max_input_len used during fine tuning is 256
    MAX_INPUT_LEN = args.max_input_length
    blackbox, class2label = load_blackbox_and_tokenizer(args.dataset, args.model_path)
    blackbox.model_to_device()
    
    # define feature attribution scorer
    FeatScorer = FetureImportanceScorer(blackbox.model, blackbox.tokenizer, method='integrated_gradient')
    
    output_path = os.path.join(args.output_path, args.dataset)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print_msg(f"Storing feature importances at: {output_path}")
    
    # extract feature importances and save on file
    for df, df_name in zip([train_df, valid_df, test_df], ['train', 'valid', 'test']):
        print_msg(f"Doing {df_name}...")
        
        #print_msg("Taking a small sample for test..")
        #df = df.iloc[:100]
        
        out_file = os.path.join(output_path, f"{df_name}_with_token_importances_wrt_{args.explain_wrt}.json")
        if os.path.exists(out_file):
            print_msg(f"The file {out_file} already exists.")
            continue
        
        # predict label of instances
        print_msg(f"Predicting labels...")
        predictions = blackbox(df.text.tolist(), show_progress=True)
        predicted_labels = predictions.argmax(axis=-1)
        predicted_proba = predictions[:, 1]
        
        df.loc[:, 'prediction'] = predicted_labels
        df.loc[:, 'pred_proba'] = predicted_proba
        print_msg(f"Labels predicted.")

        print_msg(f"Extracting feature importance...")
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            
            label = row.label if args.explain_wrt == 'original' else row.prediction
            
            # get feature importances
            text = truncate_string(row.text, blackbox.tokenizer, max_tokens=MAX_INPUT_LEN).lower()
            
            explanation = FeatScorer.explain_text(text, class_id=label)
            tokens_classifier = explanation['features']
            importances_classifier = explanation['importances']

            # get spans of each token
            tokens_classifier, tokens_classifier_importances, tokens_classifier_spans = \
                                                align_feature_importances_with_offsets(text,
                                                                                       tokens_classifier,
                                                                                       importances_classifier)

            line = {
                'id':row.id,
                'text':text,
                'label':label,
                'original_label':row.label,
                'pred_label':row.prediction,
                'pred_proba':row.pred_proba,
                'tokens_classifier':tokens_classifier,
                'tokens_classifier_importances':tokens_classifier_importances,
                'tokens_classifier_spans':tokens_classifier_spans
            }

            write_line(json.dumps(line), out_file)
                        
    print_msg("Token importance extraction ended!")



if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()    
    parser.add_argument('--data-path', type=str, help='The path of the folder containing the original dataset.')
    parser.add_argument('--model-path', type=str, help='The path of the folder containing the blackbox model.')
    parser.add_argument('--output-path', type=str, help='The path where the processed dataset will be saved.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset to processs.')
    parser.add_argument('--max-input-length', type=int, help='The maximum number of tokens for each input.')
    parser.add_argument('--explain-wrt', type=str, help='Get importances with respect to the original label or the predicted label.', choices=['original', 'predicted'])
    args = parser.parse_args()
        
    # set device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print_msg("Start extraction of feature importances.")
    print_msg(args, with_time=False)
    main(args)
