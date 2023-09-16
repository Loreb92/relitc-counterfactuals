import os
import re
import json
import glob
import shutil
import torch

from transformers import AutoTokenizer, Trainer, BatchEncoding


def padded_stack(tensors, side="right", mode="constant", value=0):
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.
    From: https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/utils.html#padded_stack

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            torch.nn.functional.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out


class DataCollatorWithPaddingForCMLM:
    """
    Data collator that will dynamically pad the inputs received.
    Slightly modified from: https://github.com/huggingface/transformers/blob/v4.22.1/src/transformers/data/data_collator.py#L213
    This is able to pad also the labels
    """
    
    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt'):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.key2padValue = {'input_ids':tokenizer.pad_token_id, 'attention_mask':0, 'token_type_ids':0, 'labels':-100}


    def __call__(self, features):
        """
        Parameters:
        features : list of dicts, dicts like {'tensor_name':1dTorchTensor} where 'tensor_name' in ['input_ids', 'attention_mask', 'token_type_ids', 'labels'] .
        """
        
        # let's convert the input list of dicts in a dict of lists
        features = {key: padded_stack([example[key] for example in features], value=self.key2padValue[key]) for key in features[0].keys()}
        
        return BatchEncoding(features)



def load_trainer_state(folder):
    """ Load last trainer state. It contains
    """

    checkpoint_folders = glob.glob(os.path.join(folder, "checkpoint-*"))
    last_checkpoint_folder = max(checkpoint_folders, key=lambda fold: int(re.findall("(?<=checkpoint-)\d+", fold)[0]))

    trainer_state = json.load(open(os.path.join(last_checkpoint_folder, "trainer_state.json")))

    return trainer_state


def train(model_path, 
          model_class, 
          train_dataset, 
          valid_dataset, 
          training_args, 
          tokenizer=None,
          data_collator_f=None,
          callbacks=None):
    """
    Train the model. After training, only the best model is saved.
    """
    
    output_dir = training_args.output_dir

    # load model
    model = model_class.from_pretrained(model_path);
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path);

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset, 
        data_collator=data_collator_f,
        callbacks=callbacks
    )

    trainer.train()

    # save the final model
    # delete all the intermediate models and save the final trainer state containing all the losses
    trainer_state = load_trainer_state(output_dir)
    shutil.rmtree(output_dir) 
    os.mkdir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    json.dump(trainer_state, open(os.path.join(output_dir, "trainer_state_last.json"), 'wt'))
