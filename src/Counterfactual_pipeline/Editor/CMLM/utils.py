import numpy as np
import torch


#### Functions to select the next token to fill 
def select_token_to_replace_ordered(mask_token_indices_dict, direction='left_to_right'):
    '''
    Given the mask tokens and the strategy to fill them, select the ones to replace with predicted tokens.
    It selects one token per instance

    left to right or right to left
    
    Parameters:
    mask_token_indices_dict : dict
    direction : str
    
    Returns:
    batch_idxs : list
    token_positions : list
    '''
    
    if direction=='left_to_right':
        batch_idxs, token_positions = zip(*[(batch_idx, min(mask_positions)) 
                                            for batch_idx, mask_positions in mask_token_indices_dict.items() if len(mask_positions)>0])
    elif direction=='right_to_left':
        batch_idxs, token_positions = zip(*[(batch_idx, max(mask_positions)) 
                                            for batch_idx, mask_positions in mask_token_indices_dict.items() if len(mask_positions)>0])
    else:
        assert False
    
    return batch_idxs, token_positions



def select_token_to_replace_confidence(mask_token_indices_dict, logits, direction='highest_first'):
    """
    For each element of the batch, the mask token selected is the one where the output probability distribution is more peaked.
    We assume the model to be more confident with the prediction when the output probability is peaked.
    This property is measured with the entropy of the distribution.
    
    Parameters:
    mask_token_indices_dict : dict
    logits : tensor
    direction : str
    
    Returns:
    batch_idxs : list
    token_positions : list
    """
    
    def _get_token_to_replace_func(direction='highest_first'):
        
        if direction=='highest_first':
            return lambda x: x.argmin()
        elif direction=='lowest_first':
            return lambda x: x.argmax()
        else:
            assert False
            
    token_to_replace_func = _get_token_to_replace_func(direction)

    batch_idxs = []
    token_positions = []
    #for i in range(batch_size):
    for i, mask_indices in mask_token_indices_dict.items():

        if len(mask_indices)>1:
            # new_token_probas : (n_masks, vocab_size)
            new_token_probas = logits[i, mask_indices].softmax(dim=-1)
            mask_token_confidence = torch.special.entr(new_token_probas).sum(dim=-1)
            
            idx_mask_most_confident = token_to_replace_func(mask_token_confidence)
            
        elif len(mask_indices)==0:
            continue
        else:
            # if only one mask, keep it
            idx_mask_most_confident = 0

        batch_idxs.append(i)
        token_positions.append(mask_indices[idx_mask_most_confident])

    return batch_idxs, token_positions
