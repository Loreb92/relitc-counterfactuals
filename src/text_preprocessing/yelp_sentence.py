import os
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

def preprocess_data(row, data_split='train'):
    
    idx = data_split+"_"+str(row.id)
    text = TreebankWordDetokenizer().detokenize(row.text.split())
    #label = 0 if row.polarity==1 else 1
    label = row.polarity
    
    new_row = {'id':idx, 'text':text, 'label':label}
    return new_row


def make_dataset(data_fold):
    
    # load train
    data_train_0 = [(0, txt) for txt in open(os.path.join(data_fold, 'sentiment.train.0')).read().split("\n")[:-1]]
    data_train_1 = [(1, txt) for txt in open(os.path.join(data_fold, 'sentiment.train.1')).read().split("\n")[:-1]]
    data_train = data_train_0 + data_train_1
    data_train = pd.DataFrame(data_train, columns=['polarity', 'text'])
    data_train.index.name = 'id'
    data_train = data_train.reset_index()

    # load valid
    data_valid_0 = [(0, txt) for txt in open(os.path.join(data_fold, 'sentiment.dev.0')).read().split("\n")[:-1]]
    data_valid_1 = [(1, txt) for txt in open(os.path.join(data_fold, 'sentiment.dev.1')).read().split("\n")[:-1]]
    data_valid = data_valid_0 + data_valid_1
    data_valid = pd.DataFrame(data_valid, columns=['polarity', 'text'])
    data_valid.index.name = 'id'
    data_valid = data_valid.reset_index()

    # load test
    data_test_0 = [(0, txt) for txt in open(os.path.join(data_fold, 'sentiment.test.0')).read().split("\n")[:-1]]
    data_test_1 = [(1, txt) for txt in open(os.path.join(data_fold, 'sentiment.test.1')).read().split("\n")[:-1]]
    data_test = data_test_0 + data_test_1
    data_test = pd.DataFrame(data_test, columns=['polarity', 'text'])
    data_test.index.name = 'id'
    data_test = data_test.reset_index()
    
    # preprocess
    data_train = data_train.apply(lambda row: preprocess_data(row, data_split='train'), axis=1)
    data_train = pd.DataFrame(data_train.tolist())

    data_valid = data_valid.apply(lambda row: preprocess_data(row, data_split='valid'), axis=1)
    data_valid = pd.DataFrame(data_valid.tolist())

    data_test = data_test.apply(lambda row: preprocess_data(row, data_split='test'), axis=1)
    data_test = pd.DataFrame(data_test.tolist())
  
    return data_train, data_valid, data_test