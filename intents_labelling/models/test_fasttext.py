import fasttext
import numpy as np, pandas as pd
from gensim.utils import simple_preprocess
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from intents_labelling.models.preprocessing import remove_punctuation, query_plus_url, get_domains, get_url_stripped
from intents_labelling.models.train_fasttext import prepare_data

import csv


def load_test(data_path: str):
    df = pd.read_csv(data_path, sep="\t")
    return df


def load_mod(data_path: str):
    model = fasttext.load_model(data_path)
    return model


def save_test(data_path,df,data,label):
    df[[data, label]].to_csv(data_path,index=False,sep=' ',header=None,quoting=csv.QUOTE_NONE,quotechar="",escapechar=" ")



def model_test(mod,file_path):
    m = mod.test(file_path)
    return m


def prediction_labels(model, df, data, label):
    test_preds = []
    for row in df[data].tolist():
        test_preds.append(model.predict(row)[0][0])
    df["test_prediction"] = test_preds
    res = df[label].to_numpy()
    pred = df["test_prediction"].to_numpy()
    t = (res, pred)
    df.to_csv("test_query_pred.csv")
    return t


if __name__ == "__main__":
    t = load_test("data/output/orcas_1005.tsv")
    g_d = get_domains(t,"url")
    t_r = remove_punctuation(g_d,"domain_names")
    q_url = query_plus_url(t_r, "query","domain_names")
    p_d = prepare_data(t,"query_url","label_manual")
    model = load_mod("intents_labelling/model_files/ftext_train_query_url_dom.bin")
    st = save_test("intents_labelling/train_test_files/test_query_url_dom.txt", p_d, "query_url", "label_manual")
    print(model_test(model, "intents_labelling/train_test_files/test_query_url_dom.txt"))
    pl = prediction_labels(model, p_d, "query_url", "label_manual")
    print(precision_recall_fscore_support(pl[0], pl[1], average="macro"))
    cl = p_d.label_manual.unique().tolist()
    print(classification_report(pl[0], pl[1], labels=cl))
    """
    model = load_mod("intents_labelling/model_files/ftext_train_query.bin")
    st= save_test("intents_labelling/train_test_files/test_query.txt", p_d, "query", "label_manual")
    print(model_test(model, "intents_labelling/train_test_files/test_query.txt"))
    pl = prediction_labels(model, p_d, "query", "label_manual")
    print(precision_recall_fscore_support(pl[0], pl[1], average="macro"))
    cl = p_d.label_manual.unique().tolist()
    print(classification_report(pl[0], pl[1], labels=cl))

    """
