import numpy as np
import torch


class MaskerDataset(torch.utils.data.Dataset):
    """
    Dataset object that masks a portion of the input texts. This can be a fraction sampled uniformy from a range or a defined fraction of tokens.
    """
    def __init__(self, df, tokenizer, id2label, use_ctrl_code=True, frac_mask_words=(0.2, 0.55), seed=None):
        """
        Parameters:
        df : pandas DataFrame, the dataframe containing the dataset. It has to have the following columns: 
            - text : str, contains the text of the instance
            - label : int, the label of the instance (note: might be either the original or predicted label)
            - tokens_classifier_importances : list of float, importance of tokens with respect to the label
            - tokens_classifier_spans : the offsets of the tokens
        tokenizer : transformers Tokenizer
        id2label : dict, dictionary mapping the label to the class name (str). It is used to add the control code. During generation, this dict maps the class to the counterfactual class name.
        use_ctrl_code : bool, whether to prepend the control code to the input text
        frac_mask_words : float or tuple, the fraction of tokens to mask. If tuple, it is the range from which the fraction is sampled
        """
        
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id
        self.use_ctrl_code = use_ctrl_code
        self.frac_mask_words = frac_mask_words
        self.RNG = np.random.RandomState(seed=seed)
        
    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        
        item = self.df.loc[idx]
        
        text = item.text
        label = item.label
        token_importances = item.tokens_classifier_importances
        token_spans = item.tokens_classifier_spans
        
        # input representation
        # if we prepend some string, we need to shift the spans accordingly
        if not self.use_ctrl_code:
            righ_shift_spans = 0
        else:
            ctrl_code = self.id2label[label].title() + " : "
            righ_shift_spans = len(ctrl_code)
            text = ctrl_code + text
        
        token_spans = [(span[0]+righ_shift_spans, span[1]+righ_shift_spans) for span in token_spans]
   
        # encode input text with spans
        encoded_input = self.tokenizer(text, truncation=True, return_offsets_mapping=True)
        offsets_mapping = encoded_input['offset_mapping']
        encoded_input = {k:torch.tensor(v, dtype=torch.long) for k, v in encoded_input.items() if k!='offset_mapping'}
            
        # make mask
        mask = self.make_mask_token_importances(offsets_mapping, token_importances, token_spans)
            
        # mask input and make labels
        encoded_input['labels'] = encoded_input['input_ids'].clone().detach()
        encoded_input['labels'][~mask] = -100
        encoded_input['input_ids'][mask] = self.mask_token_id
              
        return encoded_input
        
        
    def make_mask_token_importances(self, offsets_mapping, token_importances, token_spans):
        """
        Create mask tensor of the input text based on token importances.
        """
        
        def _range_intersection(range_1, range_2):
            return set(range(*range_1)).intersection(set(range(*range_2)))
        
        # get number of words to mask
        if type(self.frac_mask_words) is tuple:
            # it is the range where frac has to be sampled
            frac_words_to_mask = (self.frac_mask_words[1] - self.frac_mask_words[0]) * self.RNG.random() + self.frac_mask_words[0]
            n_tokens_to_mask = int(np.ceil(len(token_importances) * frac_words_to_mask))
        else:
            # it is the mask fraction
            n_tokens_to_mask = int(np.ceil(len(token_importances) * self.frac_mask_words))
        
        # get index of words to mask and corresponding spans
        idxs_to_mask = np.argsort(token_importances)[-n_tokens_to_mask:]
        spans_to_mask = [token_spans[idx] for idx in sorted(idxs_to_mask)]
        
        # make mask. Tokens to be masked correspond to 1
        # loop across all offsets in input text. Whenever one of them intersects the offset of a masked token, a mask is added to the mask list. We need to do this because of tokenizer splitting words into subwords, while we have feature importances at word level.
        mask = []
        for offset in offsets_mapping:
            lb, ub = offset
            if lb==ub: 
                # these are the special tokens
                mask.append(0)
                continue
                
            # search for intersection with masked spans
            # TODO: might be done more efficiently
            is_to_mask = False
            for span in spans_to_mask:
                if _range_intersection(span, offset): # empty set returns False
                    is_to_mask = True
                    break
                
            if is_to_mask:
                mask.append(1)
            else:
                mask.append(0)
                       
        return torch.tensor(mask, dtype=bool)
        
        