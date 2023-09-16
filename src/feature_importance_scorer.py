import numpy as np
import torch
from transformers_interpret import SequenceClassificationExplainer


class IntegratedGradientImportanceScorer():
    """
    NOTE!!! The tokens returned are different from the ones given by the only tokenizer! This is because SequenceClassificationExplainer has a text cleaning step inside!
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.scorer = SequenceClassificationExplainer(self.model, self.tokenizer, attribution_type='lig')


    def _explain_single(self, x, class_id):
        '''
        x : str, the text to explain
        class_id : int, the class to explain
        '''

        attributions = self.scorer(x, index=class_id)

        features = []
        values = []
        # discard special tokens (first and last)
        for attrib in attributions[1:-1]:
            features.append(attrib[0])
            values.append(attrib[1])

        values = np.array(values)
        
        return {'features':features, 'importances':values}


class FetureImportanceScorer():
    '''
    This class defines the object that computes feature importance of input tokens given a black box and an input text.
    
    For the moment, two methods are implemented: Integrated Gradients and Shap.
    
    NOTE! For the moment, it works surely with the transformers Bert for (binary) sequence classification.
    NOTE! Feature importances of special tokens (CLS and SEP) are not reported.
    
   
    '''
    def __init__(self, model, tokenizer, method='shap', other_args={}):
        '''
        Parameters:
        model : the black box model. For the moment, it works surely with the transformers Bert for (binary) sequence classification.
        tokenizer : tokenizer of the model. For the moment, the transformers tokenizer of Bert
        method : str, one of ['shap', 'integrated_gradient']
        other_args : dict, additional arguments that may be useful for a particular attribution method
        '''
        
        self.model = model
        self.tokenizer = tokenizer

        self.method = method
        assert method in ['shap', 'integrated_gradient'], 'The method is not implemented.'
        self.other_args = other_args

        self.build_scorer()
        

    def build_scorer(self):

        if self.method=='shap':
            self.max_evals = self.other_args['max_evals'] if 'max_evals' in self.other_args.keys() else 500 # 500 is the default valued of the library
            self.importance_scorer = ShapImportanceScorer(self.model, self.tokenizer, self.max_evals)

        elif self.method=='integrated_gradient':
            self.importance_scorer = IntegratedGradientImportanceScorer(self.model, self.tokenizer)

    def explain_text(self, text, class_id):
        '''
        Computed the feature importance of an input text with respect to the desired class.
        
        Parameters:
        text : str, the text to explain
        class_id : int in {0, 1}, the class with respect to which we want to obtain the explanations
        
        Returns: 
        explanation : dict, it contains the list of tokens and the list of feature importances
        '''

        # get the text tokens as per the model input
        text_tokenized = self.tokenizer.tokenize(text)

        # run interpreter
        interpretation = self.importance_scorer._explain_single(text, class_id)

        # truncated words should be merged
        #return {'features':interpretation['features'], 'importances':interpretation['importances']}
        #return self._merge_truncated_tokens(text_tokenized, interpretation)
        return self._merge_truncated_tokens(interpretation)
        


    @staticmethod
    #def _merge_truncated_tokens(tokens, interpretation):
    def _merge_truncated_tokens(interpretation):
        """ 
        Importance of sub-word tokens is merged through max pooling.
        """

        token_scores = interpretation['importances']
        tokens = interpretation['features']

        tokens_aggregated = []
        scores_aggregated = []
        for token, token_score in zip(tokens, token_scores):

            if token.startswith("##"):  # this is how Bert truncate words
                tokens_aggregated[-1] = tokens_aggregated[-1]+token.lstrip("##")
                scores_aggregated[-1] = max(scores_aggregated[-1], token_score)
            else:
                tokens_aggregated.append(token)
                scores_aggregated.append(token_score)

        scores_aggregated = np.array(scores_aggregated)

        return {'features':tokens_aggregated, 'importances':scores_aggregated}
