# this whole code is taken from MiCe repo
import random
import numpy as np
import more_itertools as mit
import spacy
import torch

from transformers import T5Tokenizer, T5Model, T5Config
from transformers import T5ForConditionalGeneration


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class SpacyTokenizer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=["vectors", "textcat", "parser", "ner"])
        
    def tokenize(self, text):
        return self.nlp(text)


class Masker():
    """ 
    Class used to mask inputs for Editors.
    Two subclasses: RandomMasker and GradientMasker
    
    mask_frac: float 
        Fraction of input tokens to mask.
    editor_to_wrapper: allennlp.data.tokenizers.tokenizer 
        Wraps around Editor tokenizer.
        Has capabilities for mapping Predictor tokens to Editor tokens.
    max_tokens: int
        Maximum number of tokens a masked input should have.
    """
    
    def __init__(
            self,
            mask_frac, 
            editor_tok_wrapper, 
            max_tokens
        ):
        self.mask_frac = mask_frac
        self.editor_tok_wrapper = editor_tok_wrapper
        self.max_tokens = max_tokens
        
    def _get_mask_indices(self, editor_toks):
        """ Helper function to get indices of Editor tokens to mask. """
        raise NotImplementedError("Need to implement this in subclass")

    def get_all_masked_strings(self, editable_seg):
        """ Returns a list of masked inps/targets where each inp has 
        one word replaced by a sentinel token.
        Used for calculating fluency. """

        editor_toks = self.editor_tok_wrapper.tokenize(editable_seg)
        masked_segs = [None] * len(editor_toks)
        labels = [None] * len(editor_toks)

        for i, token in enumerate(editor_toks):
            #token_start, token_end = token.idx, token.idx_end
            token_start, token_end = token.idx, token.idx + len(token)
            masked_segs[i] = editable_seg[:token_start] + \
                    Masker._get_sentinel_token(0) + editable_seg[token_end:]
            labels[i] = Masker._get_sentinel_token(0) + \
                    editable_seg[token_start:token_end] + \
                    Masker._get_sentinel_token(1)
        
        return masked_segs, labels       

    def _get_sentinel_token(idx):
        """ Helper function to get sentinel token based on given idx """

        return "<extra_id_" + str(idx) + ">"

    def _get_grouped_mask_indices(
            self, editable_seg, pred_idx, editor_mask_indices, 
            editor_toks, **kwargs):
        """ Groups consecutive mask indices.
        Applies heuristics to enable better generation:
            - If > 27 spans, mask tokens b/w neighboring spans as well.
                (See Appendix: observed degeneration after 27th sentinel token)
            - Mask max of 100 spans (since there are 100 sentinel tokens in T5)
        """

        if editor_mask_indices is None:
            editor_mask_indices = self._get_mask_indices(
                    editable_seg, editor_toks, pred_idx, **kwargs)

        new_editor_mask_indices = set(editor_mask_indices)
        grouped_editor_mask_indices = [list(group) for group in \
                mit.consecutive_groups(sorted(new_editor_mask_indices))]

        if len(grouped_editor_mask_indices) > 27:
            for t_idx in editor_mask_indices:
                if t_idx + 2 in editor_mask_indices:
                    new_editor_mask_indices.add(t_idx + 1)
        
        grouped_editor_mask_indices = [list(group) for group in \
                mit.consecutive_groups(sorted(new_editor_mask_indices))]

        if len(grouped_editor_mask_indices) > 27:
            for t_idx in editor_mask_indices:
                if t_idx + 3 in editor_mask_indices:
                    new_editor_mask_indices.add(t_idx + 1)
                    new_editor_mask_indices.add(t_idx + 2)

        new_editor_mask_indices = list(new_editor_mask_indices)
        grouped_editor_mask_indices = [list(group) for group in \
                mit.consecutive_groups(sorted(new_editor_mask_indices))]
        
        grouped_editor_mask_indices = grouped_editor_mask_indices[:99]
        return grouped_editor_mask_indices

    def get_masked_string(
            self, editable_seg, pred_idx, 
            editor_mask_indices = None, **kwargs):
        """ Gets masked string masking tokens w highest predictor gradients.
        Requires mapping predictor tokens to Editor tokens because edits are
        made on Editor tokens. """

        editor_toks = self.editor_tok_wrapper.tokenize(editable_seg)
        grpd_editor_mask_indices = self._get_grouped_mask_indices(
                editable_seg, pred_idx, editor_mask_indices, 
                editor_toks, **kwargs)
        
        span_idx = len(grpd_editor_mask_indices) - 1
        label = Masker._get_sentinel_token(len(grpd_editor_mask_indices))
        masked_seg = editable_seg

        # Iterate over spans in reverse order and mask tokens
        for span in grpd_editor_mask_indices[::-1]:

            span_char_start = editor_toks[span[0]].idx
            span_char_end = editor_toks[span[-1]].idx_end
            end_token_idx = span[-1]

            # If last span tok is last t5 tok, heuristically set char end idx
            if span_char_end is None and end_token_idx == len(editor_toks)-1:
                span_char_end = span_char_start + 1

            if not span_char_end > span_char_start:
                raise MaskError
                
            label = Masker._get_sentinel_token(span_idx) + \
                    masked_seg[span_char_start:span_char_end] + label
            masked_seg = masked_seg[:span_char_start] + \
                    Masker._get_sentinel_token(span_idx) + \
                    masked_seg[span_char_end:]
            span_idx -= 1    

        return grpd_editor_mask_indices, editor_mask_indices, masked_seg, label
            
        
class RandomMasker(Masker):
    """ Masks randomly chosen spans. """ 
    
    def __init__(
            self, 
            mask_frac, 
            editor_tok_wrapper, 
            max_tokens
        ):
        super().__init__(mask_frac, editor_tok_wrapper, max_tokens)
   
    def _get_mask_indices(self, editable_seg, editor_toks, pred_idx, **kwargs):
        """ Helper function to get indices of Editor tokens to mask. """
        
        num_tokens = min(self.max_tokens, len(editor_toks))
        return random.sample(
                range(num_tokens), math.ceil(self.mask_frac * num_tokens))



class EditEvaluator():
    def __init__(
        self,
        fluency_model_name = "t5-base",
        fluency_masker = RandomMasker(None, SpacyTokenizer(), 512) 
    ):
        self.device = get_device()
        
        if 'bert' in fluency_model_name.lower():
            self.fluency_model = BertForMaskedLM.from_pretrained(
                    fluency_model_name).to(self.device)
            self.fluency_tokenizer = BertTokenizer.from_pretrained(
                    fluency_model_name)
            self.fluency_masker = fluency_masker 
            
        else:
            self.fluency_model = T5ForConditionalGeneration.from_pretrained(
                    fluency_model_name).to(self.device)
            self.fluency_tokenizer = T5Tokenizer.from_pretrained(
                    fluency_model_name)
            self.fluency_masker = fluency_masker 
            self.fluency_model.eval()
            
            
    def score_texts(self, original_text, counterfactual_texts):
        
        scores = [self.score_fluency_new(txt) for txt in [original_text] + counterfactual_texts ]
        
        return [score/scores[0] for score in scores[1:]]
            
            
    def score_fluency_original(self, sent):
        """
        This is the same method used in: https://github.com/allenai/mice/blob/main/src/edit_finder.py#49:~:text=def%20score_fluency(self%2C%20sent)%3A
        """
        temp_losses = []
        masked_strings, span_labels = \
                self.fluency_masker.get_all_masked_strings(sent)
        for masked, label in zip(masked_strings, span_labels):
            input_ids = self.fluency_tokenizer.encode(masked, 
                    truncation="longest_first", max_length=600, 
                    return_tensors="pt")
            input_ids = input_ids.to(self.device)
            labels = self.fluency_tokenizer.encode(label, 
                    truncation="longest_first", max_length=600, 
                    return_tensors="pt")
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.fluency_model(input_ids=input_ids, labels=labels)
            loss = outputs[0] 
            temp_losses.append(loss.item())
            del input_ids
            del labels
            del loss
            torch.cuda.empty_cache()
        avg_loss = sum(temp_losses)/len(temp_losses)
        return avg_loss
            
            

    def score_fluency_new(self, sent, batch_size=32):
        """
        Same method as score_fluency_original but faster because of batching.
        """
        temp_losses = []
        masked_strings, span_labels = \
                self.fluency_masker.get_all_masked_strings(sent)
        n_masks = len(span_labels)
        
        for i in range(0, n_masks, batch_size):
            
            masked_batch = masked_strings[i:i+batch_size]
            label_batch = span_labels[i:i+batch_size]
            
            input_encoded = self.fluency_tokenizer(masked_batch, 
                    truncation="longest_first", max_length=600, padding='longest',
                    return_tensors="pt")
            input_encoded = input_encoded.to(self.device)
            labels = self.fluency_tokenizer(label_batch, 
                    truncation="longest_first", max_length=600, padding='longest',
                    return_tensors="pt")
            labels = labels['input_ids']
            labels[labels==self.fluency_tokenizer.pad_token_id] = -100 # change padding token to -100, which is not used for cross entropy
            labels = labels.to(self.device)
            
            with torch.no_grad():
                outputs = self.fluency_model(**input_encoded, labels=labels, return_dict=True)
            logits = outputs.logits.detach()        
                
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            for batch_id in range(len(label_batch)):
                logits_ = logits[batch_id].unsqueeze(0)
                labels_ = labels[batch_id].unsqueeze(0)
                loss_ = loss2 = loss_fct(logits_.view(-1, logits_.size(-1)), labels_.view(-1))
                temp_losses.append(loss_.item())
            
        avg_loss = sum(temp_losses)/len(temp_losses)
        #avg_loss = sum(temp_losses)/n_masks
        return avg_loss
    
    