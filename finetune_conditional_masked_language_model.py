import os
import argparse
import pickle
import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, EarlyStoppingCallback

from src.utils import print_msg
from src.training_transformers import train, DataCollatorWithPaddingForCMLM
from src.Counterfactual_pipeline.Editor.CMLM.masker_dataset import MaskerDataset

def add_finetuning_config_args(parser):
    
    group = parser.add_argument_group('finetune', 'finetune configurations')
    group.add_argument('--learning-rate', type=float, help="Learning rate")
    group.add_argument('--weight-decay', type=float, help="Weight decay", default=0.01)
    group.add_argument('--train-batch-size', type=int, help="Training batch size")
    group.add_argument('--eval-batch-size', type=int, help="Evaluation batch size")
    group.add_argument('--gradient_accumulation_steps', type=int, help="Gradient accumulation steps")
    group.add_argument('--train-epochs', type=int, help="Number of training epochs")
    group.add_argument('--warmup-steps', type=int, help="Number of warmup steps", default=0)
    group.add_argument('--lr-scheduler', type=str, help="Learning rate scheduler type")
    group.add_argument('--patience', type=int, help="Early stopping patience")
    
    group.add_argument('--logging-strategy', type=str, help="Logging frequency", default='steps')
    group.add_argument('--logging-steps', type=int, help="Number of steps between two logs", default=50)
    group.add_argument('--save-strategy', type=str, help="Save frequency")
    group.add_argument('--save-steps', type=int, help="Number of steps between two model checkpoint save")
    group.add_argument('--save-total-limit', type=int, help="Max number of checkpoints to save", default=4)
    group.add_argument('--evaluation-strategy', type=str, help="Evaluation frequency")
    group.add_argument('--evaluation-steps', type=int, help="Number of steps between two evaluations")
    group.add_argument('--load-best-model-at-end', type=bool, help="Whether to load the best model", default=True)
    group.add_argument('--greater-is-better', type=bool, help="Whether the best validation score has to be the maximum or minimum", default=False)
    
    return parser


def load_data(dataset, data_path, explain_wrt):
    
    data_fold = os.path.join(data_path, dataset)

    train_df = pd.read_json(os.path.join(data_fold, f"train_with_token_importances_wrt_{explain_wrt}.json"),
                            orient='records', lines=True)
    valid_df = pd.read_json(os.path.join(data_fold, f"valid_with_token_importances_wrt_{explain_wrt}.json"),
                            orient='records', lines=True)

    return train_df, valid_df
    
    
def make_datasets(train_df, valid_df, tokenizer, id2label, use_ctrl_code=None, frac_mask_words=None, seed=None):
    """
    Created the datasets.
    """
    
    train_dataset = MaskerDataset(train_df, tokenizer, id2label, use_ctrl_code=use_ctrl_code, frac_mask_words=frac_mask_words, seed=seed)
    valid_dataset = MaskerDataset(valid_df, tokenizer, id2label, use_ctrl_code=use_ctrl_code, frac_mask_words=frac_mask_words, seed=seed)

    return train_dataset, valid_dataset


def get_label_name_dict(dataset):
    
    if dataset == 'Yelp':
        return {0 : 'negative', 1 : 'positive'}
    elif dataset == 'OLID':
        return {0 : 'not offensive', 1 : 'offensive'}
    elif dataset == 'yelp_sentence':
        return {0 : 'negative', 1 : 'positive'}
    elif dataset == 'call_me':
        return {0 : 'non-sexist', 1 : 'sexist'}
    else:
        raise NotImplementedError(f"Scripts for dataset {dataset} are not implemented.")
    
    
def main(args):
    
    # load data
    print_msg("Loading data...")
    train_df, valid_df = load_data(args.dataset, args.data_path, args.explain_wrt)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # define datasets
    print_msg("Making datasets...")
    id2label = get_label_name_dict(args.dataset)
    train_dataset, valid_dataset = make_datasets(train_df, valid_df, tokenizer, id2label, use_ctrl_code=True, frac_mask_words=(args.min_mask_frac, args.max_mask_frac), seed=args.seed)
    
    # eval 4 times per epoch
    n_steps_per_epoch = train_dataset.__len__() // args.train_batch_size
    save_every_steps = int(np.ceil(n_steps_per_epoch / 4))

    # define training arguments and train
    model_output_dir = os.path.join(args.model_output_dir, args.dataset)
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    model_output_dir = os.path.join(model_output_dir, f"finetuned_CMLM_maks_frac-{args.min_mask_frac}_{args.max_mask_frac}-explain_wrt-{args.explain_wrt}")
    
    print_msg(f"Saving model checkpoints at: {model_output_dir}")
    if os.path.exists(model_output_dir):
        print_msg("This experiment had already been performed. Do not overwrite results.")
        return 1
    
    training_args = TrainingArguments(
                                    do_train=True,
                                    output_dir=model_output_dir,
                                    learning_rate=args.learning_rate,
                                    weight_decay=args.weight_decay,
                                    per_device_train_batch_size=args.train_batch_size,
                                    per_device_eval_batch_size=args.eval_batch_size,
                                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                                    num_train_epochs=args.train_epochs,
                                    warmup_steps=args.warmup_steps,
                                    lr_scheduler_type=args.lr_scheduler,
                                    logging_strategy=args.logging_strategy,
                                    logging_steps=args.logging_steps,
                                    save_strategy=args.save_strategy,
                                    save_steps=save_every_steps, # each half epoch
                                    save_total_limit=args.save_total_limit,
                                    seed=args.seed,
                                    evaluation_strategy=args.evaluation_strategy,
                                    eval_steps=save_every_steps,
                                    load_best_model_at_end=args.load_best_model_at_end,
                                    greater_is_better=args.greater_is_better
                                )
    
    callbacks = []
    if args.patience and args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    print_msg("Start training...")
    train(args.model_dir, 
          model_class=AutoModelForMaskedLM, 
          tokenizer=tokenizer,
          train_dataset=train_dataset, 
          valid_dataset=valid_dataset, 
          training_args=training_args, 
          data_collator_f=DataCollatorWithPaddingForCMLM(tokenizer, padding=True), 
          callbacks=callbacks)
          
    pickle.dump(args, open(os.path.join(model_output_dir, 'args.pkl'), "wb"))

    print_msg("Training finished!")
    

if __name__ == '__main__':
        
    # parse arguments
    parser = argparse.ArgumentParser()    
    parser.add_argument('--data-path', type=str, help='The path of the folder containing the processed dataset.')
    parser.add_argument('--dataset', type=str, help='The name of the dataset to processs.')
    parser.add_argument('--explain-wrt', type=str, help='Get importances with respect to the original label or the predicted label.', choices=['original', 'predicted'])
    parser.add_argument('--seed', type=int, help='Random number generators seed', default=42)
    
    parser.add_argument('--model-path', type=str, help='The path of the folder containing the pretrained model.')
    parser.add_argument('--model-output-dir', type=str, help='The path where the fine tuned models are saved.')
    parser.add_argument('--model-name', type=str, help='The name of the pretrained model', default='bert-base-uncased')
    parser.add_argument('--min-mask-frac', type=float, help='Minimum token mask fraction', default=0.20)
    parser.add_argument('--max-mask-frac', type=float, help='Maximum token mask fraction', default=0.55)
    
    add_finetuning_config_args(parser)
    args = parser.parse_args()
    
    if args.model_name in ['bert-base-uncased']:
        # if the pretrained model is in the Hugging Face Hub
        args.model_dir = args.model_name
    else:
        # if the pretrained model is stored locally
        args.model_dir = os.path.join(args.model_path, args.model_name)
        
    # set device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print_msg(args, with_time=False)
    main(args)
    
    
    