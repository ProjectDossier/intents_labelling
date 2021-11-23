import fasttext
import numpy as np, pandas as pd
from gensim.utils import simple_preprocess
import csv
import re


def query_to_url(df):
    """
    preprocesses the query into query and url
    """
    df["query"] = df["query"] = "query "+df["query"]+" url "+df["url"]
    return df

def query_domain(df):
    """
    gets the domain names for the queries
    """
    l = df["url"].to_list()
    l_stripped = []
    for el in l:
        r1 = re.search(r"https:\/\/www\.(.*?)\/", el)
        r2 = re.search(r"http:\/\/www\.(.*?)\/", el)
        r3 = re.search(r"http:\/\/(.*?)\/", el)
        r4 = re.search(r"https:\/\/(.*?)\/", el)
        st = ""
        if r1:
            st = r1.group(1)
            l_stripped.append(st)
        elif r2:
            st = r2.group(1)
            l_stripped.append(st)
        elif r3:
            st = r3.group(1)
            l_stripped.append(st)
            print("el", el)
        elif r4:
            st = r4.group(1)
            l_stripped.append(st)
        else:
            l_stripped.append(el)
    df["url"] = l_stripped
    return df

def prepare_train(df):
    """
    puts the data into fast text format
    """
    df = df[["query", "Label"]]
    df = df.fillna('')
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: ' '.join(simple_preprocess(x)))
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: '__label__' + x)
    return df

def model_train(df,data_path):
    """
    train the model
    """
    df[['query', 'Label']].to_csv(data_path,index=False,sep=' ',header=None,quoting=csv.QUOTE_NONE,quotechar="",escapechar=" ")
    model = fasttext.train_supervised(data_path, wordNgrams=2)
    return model

def save_mod(model,data_path):
    model.save_model(data_path)

if __name__ == "__main__":
    dataset = pd.read_csv('data/output/orcas_1000000.tsv', sep="\t")
    l_stripped = query_domain(dataset)
    print(l_stripped["url"])
    ds_train = dataset.sample(frac=0.8,random_state=10)
    ds_test = dataset[~dataset.index.isin(ds_train.index)]
    df2 = prepare_train(ds_train)
    model = model_train(df2,"intents_labelling/train_test_files/train.txt")
    save_mod(model,"intents_labelling/model_files/ftext_train.bin")




