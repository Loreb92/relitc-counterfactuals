import unicodedata
import re
import os
import pandas as pd

def unescape_characters(text):
    return text.encode().decode('unicode-escape')

def remove_newlines(text):
    
    # remove newlines and remove multiple spaces
    text = re.sub("\n", " ", text)
    text = re.sub("\s{2,}", " ", text)
    return text


def preprocess_data(row, data_split='train'):
    
    idx = data_split+"_"+str(row.id)
    
    # remove accents and non-latin characters
    text = unescape_characters(row.text)
    text = unicodedata.normalize('NFD', text)\
                                       .encode('ascii', 'ignore')\
                                       .decode("utf-8")
    text = remove_newlines(text)
    
    label = 0 if row.polarity==1 else 1
    
    new_row = {'id':idx, 'text':text, 'label':label}
    return new_row


def make_dataset(data_fold):
    
    # load dataset
    train_df = pd.read_csv(os.path.join(data_fold, 'yelp_review_polarity_csv', 'train.csv'), header=None).reset_index()
    train_df.columns = ['id', 'polarity', 'text']

    test_df = pd.read_csv(os.path.join(data_fold, 'yelp_review_polarity_csv', 'test.csv'), header=None).reset_index()
    test_df.columns = ['id', 'polarity', 'text']

    # sample 100K instances in total
    train_df = train_df.groupby('polarity').sample(50000, random_state=42).sample(frac=1, random_state=42)
    train_df = train_df.reset_index(drop=True)

    # sample 10K for test
    test_df = test_df.groupby('polarity').sample(5000, random_state=42).sample(frac=1, random_state=42)
    test_df = test_df.reset_index(drop=True)

    # split train/valid
    val_frac = 0.2

    valid_df = train_df.groupby('polarity').sample(frac=0.2, random_state=42).sample(frac=1, random_state=42)
    train_df = train_df[~train_df.id.isin(valid_df.id)]

    # data cleaning
    train_df = train_df.apply(lambda row: preprocess_data(row, data_split='train'), axis=1)
    train_df = pd.DataFrame(train_df.tolist())

    valid_df = valid_df.apply(lambda row: preprocess_data(row, data_split='valid'), axis=1)
    valid_df = pd.DataFrame(valid_df.tolist())

    test_df = test_df.apply(lambda row: preprocess_data(row, data_split='test'), axis=1)
    test_df = pd.DataFrame(test_df.tolist())
    
    
    return train_df, valid_df, test_df