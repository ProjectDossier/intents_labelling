import fasttext
import numpy as np, pandas as pd
from gensim.utils import simple_preprocess
import csv
import re


def query_plus_url(df, data, url):

    """
    preprocesses the query into query and url
    """
    df["query_url"] = "query " + df["query"] + " url " + df[url]
    return df


def url_strip(df):
    """
    removes http and www from the url
    """
    df["url_strip"] = df["url"].apply(
        lambda x: re.sub(
            r"(https:\/\/www\.)|(http:\/\/www\.)|(http:\/\/)|(https:\/\/)", "", str(x)
        )
    )
    return df


def url_domain(df):
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
        elif r4:
            st = r4.group(1)
            l_stripped.append(st)
        else:
            l_stripped.append(el)
    df["url_domains"] = l_stripped
    return df


def prepare_data(df,data,label):
    """puts the data into fast text format"""
    df = df[[data, label]]
    df = df.fillna("")
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: " ".join(simple_preprocess(x)))
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: "__label__" + x)
    return df


def model_train(df,data_path,data,label):
    """
    train the model
    """
    df[[data, label]].to_csv(
        data_path,
        index=False,
        sep=" ",
        header=None,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ",
    )
    model = fasttext.train_supervised(data_path, wordNgrams=2)
    return model


def save_mod(mod, data_path):
    mod.save_model(data_path)


if __name__ == "__main__":
    dataset = pd.read_csv("data/output/orcas_1000000.tsv", sep="\t")
    url_st = url_strip(dataset)
    print("url st", url_st["url"])
    url_st.to_csv("strip.csv", index=False)
    ds_train = dataset.sample(frac=0.8, random_state=10)
    ds_test = dataset[~dataset.index.isin(ds_train.index)]
    df2 = prepare_data(ds_train,"query","Label")
    model = model_train(df2, "intents_labelling/train_test_files/train.txt","query","Label")
    save_mod(model, "intents_labelling/model_files/ftext_train.bin")
