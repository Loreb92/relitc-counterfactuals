import unicodedata
import html
import pandas as pd
import os


def unescape_characters(text):
    return text.encode().decode('unicode-escape')


def remove_mentions_and_urls(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def preprocess_data(row, data_split='train'):
    
    idx = data_split+"_"+str(row.id)

    # remove accents and non-latin characters
    text = unescape_characters(row.tweet)
    text = html.unescape(text)
    text = remove_mentions_and_urls(text)
    text = unicodedata.normalize('NFD', text)\
                                       .encode('ascii', 'ignore')\
                                       .decode("utf-8")

    label = 0 if row.subtask_a=='NOT' else 1
    
    new_row = {'id':idx, 'text':text, 'label':label}
    return new_row


def make_dataset(data_fold):
    
    # load data (train)
    train_df = pd.read_csv(os.path.join(data_fold, 'olid-training-v1.0.tsv'), sep='\t',
                          usecols=['id', 'tweet', 'subtask_a'])

    # load data (test)
    test_df = pd.read_csv(os.path.join(data_fold, 'testset-levela.tsv'), sep='\t')
    test_labels = pd.read_csv(os.path.join(data_fold, 'labels-levela.csv'), header=None)
    test_labels.columns = ['id', 'subtask_a']
    test_df = test_df.merge(test_labels, on='id')

    # make the dataset
    train_df = train_df.apply(lambda row: preprocess_data(row, data_split='train'), axis=1)
    train_df = pd.DataFrame(train_df.tolist())

    test_df = test_df.apply(lambda row: preprocess_data(row, data_split='test'), axis=1)
    test_df = pd.DataFrame(test_df.tolist())
    
    # split train/valid
    val_frac = 0.2

    valid_df = train_df.groupby('label').sample(frac=0.2, random_state=42).sample(frac=1, random_state=42)
    train_df = train_df[~train_df.id.isin(valid_df.id)]
    
    return train_df, valid_df, test_df