import fasttext
import numpy as np, pandas as pd
from gensim.utils import simple_preprocess
import csv
import re
from intents_labelling.models.preprocessing import remove_punctuation, query_plus_url, get_domains, get_url_stripped

def prepare_data(df,data,label):
    """puts the data into fast text format"""
    df = df[[data, label]]
    df = df.fillna("")
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: "__label__" + x)
    return df


def model_train(df,data_path,data,label):
    """
    train the model
    """
    df[[data, label]].to_csv(data_path, index=False, sep=" ", header=None, quoting=csv.QUOTE_NONE, quotechar="",escapechar=" ",)
    model = fasttext.train_supervised(data_path, wordNgrams=2)
    return model


def save_mod(mod, data_path):
    mod.save_model(data_path)


if __name__ == "__main__":

    dataset = pd.read_csv("data/output/orcas_1000000.tsv", sep="\t")
    d_train = dataset[dataset["data_type"] == "train"]
    """
    d_prep = prepare_data(d_train, "query","Label")
    model = model_train(d_prep, "intents_labelling/train_test_files/train_query.txt", "query", "Label")
    save_mod(model, "intents_labelling/model_files/ftext_train_query.bin")
    """
    g_d = get_domains(d_train,"url")
    d_p = remove_punctuation(g_d, "domain_names")
    q_url = query_plus_url(d_p,"query","domain_names")
    #q_url.to_clipboard(sep=',', index=False)
    d_prep = prepare_data(q_url, "query_url", "Label")
    model = model_train(d_prep, "intents_labelling/train_test_files/train_query_url_dom.txt", "query_url", "Label")
    save_mod(model, "intents_labelling/model_files/ftext_train_query_url_dom.bin")
