import pandas as pd
import os
import unicodedata

def preprocess_data(row, data_split='train'):
    
    idx = data_split+"_"+str(row.id)
    text = row.text
    label = row.sexist
    
    text = unicodedata.normalize('NFD', text)\
                                       .encode('ascii', 'ignore')\
                                       .decode("utf-8")
    
    new_row = {'id':idx, 'text':text, 'label':label}
    return new_row


def make_dataset(data_fold):
    
    # load data (train)
    train_df = pd.read_csv(os.path.join(data_fold, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(data_fold, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(data_fold, 'test.csv'))
    
    # make the dataset
    train_df = train_df.apply(lambda row: preprocess_data(row, data_split='train'), axis=1)
    train_df = pd.DataFrame(train_df.tolist())
    
    valid_df = valid_df.apply(lambda row: preprocess_data(row, data_split='train'), axis=1)
    valid_df = pd.DataFrame(valid_df.tolist())

    test_df = test_df.apply(lambda row: preprocess_data(row, data_split='test'), axis=1)
    test_df = pd.DataFrame(test_df.tolist())
    
    return train_df, valid_df, test_df