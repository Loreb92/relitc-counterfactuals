# https://github.com/huggingface/transformers/issues/3609

from collections import defaultdict
#from scipy.stats import entropy
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding

from ....utils import print_msg
from .utils import select_token_to_replace_ordered, select_token_to_replace_confidence



#### Class that select new token from output logits ####
class MaskInfillerToppTopk:
    def __init__(self, model, tokenizer, top_p=1.0, top_k=0, min_tokens_to_keep=1, infill_order='ordered', direction=None, seed=None):
        """
        model : 
        tokenizer : 
        top_k : int
        top_p : float
        min_tokens_to_keep : int, the minimum number of tokens to keep with nucleous sampling
        infill_order : str, in ['ordered', 'confidence']
        direction : str, in ['left_to_right', 'right_to_left'] if infill_order is 'ordered', in ['highest_first', 'lowest_first'] if infill_order is 'confidence'
        """
        
        self.model = model
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        assert infill_order in ['ordered', 'confidence'], f"Infilling order {infill_order} not implemented."        
        self.infill_order = infill_order
        
        if infill_order=='ordered':
            assert direction in ['left_to_right', 'right_to_left'] or direction is None, f"Direction {direction} with infilling order {infill_order} not implemented."
            self.direction = 'left_to_right' if direction is None else direction
        elif infill_order=='confidence':
            assert direction in ['highest_first', 'lowest_first'] or direction is None, f"Direction {direction} with infilling order {infill_order} not implemented."
            self.direction = 'highest_first' if direction is None else direction
        else:
            raise NotImplementedError(f"The infill strategy {infill_order} is not implemented.")
            
        print_msg(f"Setting editor with infill order '{self.infill_order}' and direction '{self.direction}'...")
        
        self.device = model.device
        self.mask_token_id = self.tokenizer.mask_token_id
        
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)

        
    def __call__(self, masked_inputs):
        '''
        Replace the masks with new tokens.
        This works for greedy replacement (assign only one token to a mask).

        Parameters:
        masked_inputs : (batch of masked inputs)

        Returns:
        input_ids : tensor, the encoded texts without mask tokens
        '''

        # identify mask tokens for each instance in the batch
        # inputs['input_ids']: (batch_size , n_tokens)
        # mask_token_indices: (n_masks, 2), the coordinate of the masks in inputs['input_ids']
        mask_token_indices = (masked_inputs['input_ids'] == self.mask_token_id).nonzero(as_tuple=False)
        
        # create a dict of lists, keys are batch_idx and values the location of the mask corresponding to batch_idx
        mask_token_indices_dict = defaultdict(list)
        for i, j in mask_token_indices:
            mask_token_indices_dict[i.item()].append(j.item())
        
        # if no masks, then no forward pass
        still_masks = any([len(idx_masks)>0 for idx_masks in mask_token_indices_dict.values()])
        with torch.no_grad():
            while still_masks:

                # forward pass
                # logits: (batch_size , n_tokens, vocab_size)
                with torch.no_grad():
                    outputs = self.model(**masked_inputs)
                logits = outputs['logits'].detach()
                
                # select next token to infill based on the chosen infill_order
                if self.infill_order=='ordered':
                    batch_idxs, token_positions = select_token_to_replace_ordered(mask_token_indices_dict, direction=self.direction)
                elif self.infill_order=='confidence':
                    batch_idxs, token_positions = select_token_to_replace_confidence(mask_token_indices_dict, logits, direction=self.direction)
                

                # for each instance, take one token position
                #  logits : (batch_size, vocab_size)
                logits = logits[batch_idxs, token_positions, :]

                # filter tokens based on the top_k and top_p heuristics
                logits = self._top_k_top_p_filtering(logits)

                # get the tokens with their probabilities
                #  these are the sampling probabilities for the tokens
                tokens_proba = logits.softmax(dim=-1)
                # sample tokens given the probabilities
                # sampled_token_idxs : (batch_size, num_samples)
                sampled_token_idxs = tokens_proba.multinomial(num_samples=1)
                
                # replace tokens in masked_inputs['input_ids']
                masked_inputs['input_ids'][batch_idxs, token_positions] = sampled_token_idxs.flatten()
                
                # remove mask_idxs from mask_token_indices_dict
                for batch_idx, token_position in zip(batch_idxs, token_positions):
                    mask_token_indices_dict[batch_idx].remove(token_position)
                    
                # check if still other masks
                if not torch.any(masked_inputs['input_ids'] == self.tokenizer.mask_token_id, dim=-1).any():
                    still_masks = False

        return masked_inputs['input_ids']

    
    def _top_k_top_p_filtering(self, tokens_logits):
        '''
        Selects the token(s) given the probability distribution across the vocabulary.

        (some parts of the code borrowed from https://github.com/huggingface/transformers/blob/05fa1a7ac17bb7aa07b9e0c1e138ecb31a28bbfe/src/transformers/generation_logits_process.py#L172)

        Parameters:
        tokens_logits : np.array, 2D array with dimension batch_size x vocab_size. Rows are output logits

        Returns:
        tokens_logits : np.array, 2D array
        '''
        
        if self.top_k > 0:
            # indices whose values are lower than the min of the topk values of each row
            indices_to_remove_top_k = tokens_logits < torch.topk(tokens_logits, self.top_k, dim=-1, sorted=True)[0][:,
                                                      -1, None]
            tokens_logits = tokens_logits.masked_fill(indices_to_remove_top_k, -torch.inf)

        if self.top_p < 1.:
            sorted_logits, sorted_indices = torch.sort(tokens_logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p

            if self.min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove_top_p = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

            tokens_logits = tokens_logits.masked_fill(indices_to_remove_top_p, -torch.inf)

        return tokens_logits


#### EDITOR ####
class EditorTransformersForCMLM():
    """
    Editor model for conditional masked language model.
    """
    def __init__(self, 
                 model_name,
                 sampling=False,
                 top_p=1.0, 
                 top_k=0,
                 min_tokens_to_keep=1,
                 next_token_strategy='ordered',
                 direction=None, 
                 seed=None):
        """
        Parameters:
        model_name : str
        sampling : bool
        top_p : float
        top_k : int
        min_tokens_to_keep : int
        next_token_strategy : str
        direction : str
        """
        
        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.top_p = top_p
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.direction = direction
        self.next_token_strategy = next_token_strategy
        self.device = self.model.device
        self.collate_fn = DataCollatorWithPadding(self.tokenizer, padding=True)
        
        if sampling:
            self.filling_strategy = MaskInfillerToppTopk(self.model, 
                                                         self.tokenizer,
                                                         self.top_p, self.top_k,
                                                         self.min_tokens_to_keep,
                                                         self.next_token_strategy,
                                                         self.direction, 
                                                         seed)
        else:
            raise NotImplementedError("Strategies different from sampling not implemented.")
            
            

    def model_to_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.device = self.model.device

    def model_to_cpu(self):
        self.model.to('cpu')
        self.device = self.model.device

    def __call__(self, encoded_masked_texts, batch_size=32):

        encoded_filled_texts = []
        for i in range(0, len(encoded_masked_texts), batch_size):
            # take batch
            encoded_masked_texts_batch = encoded_masked_texts[i: i + batch_size]

            # pad the batch and send to device
            encoded_masked_texts_batch = self.collate_fn(encoded_masked_texts_batch)
            encoded_masked_texts_batch.to(self.device)
            
            encoded_texts_batch = self.filling_strategy(encoded_masked_texts_batch).cpu().tolist()

            encoded_filled_texts.extend(encoded_texts_batch)

        return encoded_filled_texts
