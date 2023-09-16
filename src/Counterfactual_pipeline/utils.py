import torch
from transformers import default_data_collator

def unique_with_indices(x, dim=-1):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)



def padding_data_collator(instances, tokenizer, max_model_len=512, padding=True):
    
    """
    Pad a batch of encoded instances.
    
    Parameters:
    instances : list of dict, elements represent encoded texts
    tokenizer : transformers Tokenizer
    max_model_len : int, the maximum input length
    padding : str or bool, default True meaning pad to the longest sequence in the batch
    
    Reeturns:
    batch : transformers BatchEncoding
    """
    
    
    batch = tokenizer.pad(
            instances,
            padding=self.padding,
            max_length=max_model_len,
            pad_to_multiple_of=None,
            return_tensors='pt',
        )
    
    if "label" in batch:
        batch["labels"] = batch["label"]
        del batch["label"]
    if "label_ids" in batch:
        batch["labels"] = batch["label_ids"]
        del batch["label_ids"]
        
    return batch
    
    
    



"""
# define data collator (from instance to batch)
# it is needed to pad tensors
def padding_data_collator(instances, tokenizer, max_model_len=512):

    max_len = max([instance['input_ids'].size(0) for instance in instances])
    max_len = min(max_len, max_model_len)

    for instance in instances:

        instance['input_ids'] = pad_1D_tensor(instance['input_ids'], max_len, tokenizer.pad_token_id)
        instance['attention_mask'] = pad_1D_tensor(instance['attention_mask'], max_len, 0)
        instance['token_type_ids'] = pad_1D_tensor(instance['token_type_ids'], max_len, 0)
        if 'labels' in instance.keys():
            instance['labels'] = pad_1D_tensor(instance['labels'], max_len, -100)

    return default_data_collator(instances)


def pad_1D_tensor(a, desired_len, val):

    len_tensor = a.size(0)
    values_to_add = desired_len - len_tensor

    if values_to_add>0:
        padding = torch.ones(desired_len - len_tensor, dtype=torch.long) * val
        a = torch.hstack((a, padding))

    return a
    
"""