import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BlackBoxTransformersForSequenceClassification():
    '''
    This is the basic class for models following the Huggingface transformers library (models for sequence classification).
    '''
    def __init__(self, model_name, no_cuda=False, label_names={}):
        '''
        Parameters:
        model_name : str, the name of the model to be downloaded or the path where it is stored
        no_cuda : bool, if True avoid to use the GPU
        label_names : dict, the meaning of each output label
        '''

        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_names = label_names

        self.device = 'cuda' if torch.cuda.is_available() and not no_cuda else 'cpu'

    def model_to_device(self):
        self.model.to(self.device)

    def model_to_cpu(self):
        self.model.to('cpu')


    def __call__(self, texts, batch_size=32, show_progress=False):
        '''
        Parameters:
        texts : list of str, the input texts
        batch_size : int, the batch size to use for the prediction (default 32)

        Returns:
        out_probas : np.array, 2D array with dimension len(texts) x n_labels containing the predicted probabilities of the model.
        '''

        with torch.no_grad():

            out_probabs = []
            total = len(texts)//batch_size + int(len(texts)%batch_size == 0)
            for i in tqdm(range(0, len(texts), batch_size), total=total, disable=not show_progress):

                # make the batch and and encode it
                texts_batch = texts[i: i+batch_size]
                encoded_texts_batch = self.tokenizer(texts_batch,
                                                padding='longest',
                                                truncation=True,
                                                return_tensors='pt' )
                encoded_texts_batch = {k:val.to(self.device) for k, val in encoded_texts_batch.items()}

                # prediction
                with torch.no_grad():
                    outputs_ = self.model(**encoded_texts_batch)['logits']
                out_probas_ = torch.nn.functional.softmax(outputs_, dim=1)

                out_probabs.append(out_probas_.cpu().numpy())

        out_probabs = np.vstack(out_probabs)

        return out_probabs