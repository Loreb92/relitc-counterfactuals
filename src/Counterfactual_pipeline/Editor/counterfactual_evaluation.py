from typing import List, Optional
import spacy
import nltk

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import default_data_collator, BatchEncoding
from sentence_transformers import SentenceTransformer, util


class MaskedLanguageModelScorer():
    """
    Method from the paper "Masked language model scoring" (https://arxiv.org/abs/1910.14659).
    It is used to compute the fluency of the counterfactual.
    """
    
    def __init__(self, model_name, batch_size=32):
        """
        Parameters:
        model_name : str
        batch_size : int
        """
        
        self.model_name = model_name
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        
        self.vocab_size = self.tokenizer.vocab_size
        self.vocab_size = self.vocab_size + 1 if 'bertweet' in model_name else self.vocab_size
        
        self.device = self.model.device
        
        self.model.eval()
        
                
    def model_to_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.device = self.model.device

    def model_to_cpu(self):
        self.model.to('cpu')
        self.device = self.model.device
        
    def score_texts(self, original_text, counterfactual_texts):
        """
        Evaluate the masked language model scoring divided by the number of tokens.
        
        Parameters:
        texts: list
        
        Returns:
        text_loss : list
        """
        
        text_loss = []
        for text in [original_text] + counterfactual_texts:
            
            encoded_texts_masked, labels = self.get_all_maksed_inputs(text)
            batch_loss, n_words = 0, 0
            temp_losses = [] 
            for n in range(0, len(encoded_texts_masked), self.batch_size):
                
                encoded_texts_masked_, labels_ = encoded_texts_masked[n:n+self.batch_size], labels[n:n+self.batch_size]
                
                batch = {
                    'input_ids':encoded_texts_masked_,
                    'labels':labels_,
                    'token_type_ids':torch.zeros_like(encoded_texts_masked_, dtype=torch.long, device=self.device),
                    'attention_mask':torch.ones_like(encoded_texts_masked_, dtype=torch.long, device=self.device)
                }
                
                batch_size = len(encoded_texts_masked_)
                n_words += batch_size
                
                with torch.no_grad():
                    output = self.model(**batch)
                    
                for batch_idx in range(batch_size):
                    logits_ = output.logits[batch_idx].unsqueeze(0)
                    labels_ = batch['labels'][batch_idx].unsqueeze(0)
                    loss_ = torch.nn.functional.cross_entropy(logits_.view(-1, logits_.size(-1)), 
                                                                      labels_.view(-1))
                    temp_losses.append(loss_.item())
                
                
            #loss = batch_loss / n_words 
            loss = sum(temp_losses)/len(temp_losses)
            text_loss.append(loss)
            
        return [loss/text_loss[0] for loss in text_loss[1:] ]
                     
    
    def get_all_maksed_inputs(self, text):
    
        encoded_txt = self.tokenizer(text, truncation=True, return_tensors='pt')['input_ids'].to(self.device)

        # duplicate it as many times as the numberr of tokens
        encoded_txt = encoded_txt.repeat((encoded_txt.size(1),1))

        # create labels
        labels_ = torch.ones_like(encoded_txt, device=self.device) * -100
        mask = torch.diag(torch.ones_like(encoded_txt.diag()))
        labels = mask*torch.diag(encoded_txt) + ((1. - mask)*labels_)

        # mask encoded texts
        encoded_txt.fill_diagonal_(fill_value=self.tokenizer.mask_token_id)

        # remove sos and eos tokens
        encoded_txt, labels = encoded_txt[1:-1], labels[1:-1].to(torch.long)

        return encoded_txt, labels
    
    
class SemanticSimilarityScorer:
    """
    Computes the cosine similarity of the counterfactuals against the original input text.
    """
    
    def __init__(self, model_name, batch_size=32):
        """
        Parameters:
        model_name : str
        batch_size : int
        """

        self.sentence_encoder = SentenceTransformer(model_name)
        self.batch_size = batch_size
        
    def score_texts(self, original_text, counterfactual_texts):
        """
        Computes the cosine similarity between the embedding of original_text and all the embeddings of counterfactual texts.
       
        Parameters:
        original_text : str
        counterfactual_texts : list of str
        
        Returns:
        scores : list, contains the semantic similarity of the original_text against all the counterfactual_texts
        """
        
        # embed all and compute cosine similarities
        embeddings = self.sentence_encoder.encode([original_text]+counterfactual_texts, batch_size=self.batch_size)
        cos_similarities = util.cos_sim(embeddings, embeddings)
        
        return cos_similarities[0, 1:].tolist()
        
        


class MiceMinimalityScorer():
    def __init__(self):
        """
        Computes the minimality of counterfactuals
        
        Code borrowed from:     https://github.com/allenai/mice/blob/main/src/edit_finder.py#:~:text=def%20score_minimality(self%2C%20orig_sent%2C%20edited_sent%2C%20normalized%3DTrue)%3A
        https://github.com/allenai/allennlp/blob/main/allennlp/data/tokenizers/spacy_tokenizer.py
        """
        self.nlp = spacy.load('en_core_web_sm', disable=["vectors", "textcat", "parser", "ner"])

    def tokenize(self, text: str) -> List[spacy.tokens.Token]:
        # This works because our Token class matches spacy's.
        return self._remove_spaces(self.nlp(text))
    
    @staticmethod
    def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
        return [token for token in tokens if not token.is_space]
    
    def score_minimality(self, orig_sent, edited_sent, normalized=True):
        
        tokenized_original = [t.text for t in self.tokenize(orig_sent)]
        tokenized_edited = [t.text for t in self.tokenize(edited_sent)]
        
        lev = nltk.edit_distance(tokenized_original, tokenized_edited)
        if normalized: 
            return lev/len(tokenized_original)
        else:
            return lev

        
    def score_texts(self, original_text, counterfactual_texts):
        """
        Computes the minimality of all the counterfactual texts against the original text.
        
        Parameters:
        original_text : str
        counterfactual_texts : list of str
        
        Returns:
        minimalities : list, the minimality of all the counterfactual_texts
        """
 
        return [self.score_minimality(original_text, counterfactual_text) for counterfactual_text in counterfactual_texts]
        
        