import fasttext
import numpy as np, pandas as pd
from gensim.utils import simple_preprocess
from intents_labelling.models.train_fasttext import query_to_url
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

import csv


def load_test(data_path: str):
    df = pd.read_csv(data_path, sep="\t")
    return df


def load_mod(data_path: str):
    model = fasttext.load_model(data_path)
    return model


def save_test(data_path, df, data, label):
    df[[data, label]].to_csv(
        data_path,
        index=False,
        sep=" ",
        header=None,
        quoting=csv.QUOTE_NONE,
        quotechar="",
        escapechar=" ",
    )


def prepare_test(df, data, label):
    df = df[[data, label]]
    df = df.fillna("")
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: " ".join(simple_preprocess(x)))
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: "__label__" + x)
    return df


def model_test(file_path):
    m = model.test(file_path)
    return m


def prediction_labels(model, df, data, label):
    test_preds = []
    for row in df[data].tolist():
        test_preds.append(model.predict(row)[0][0])
    df["test_prediction"] = test_preds
    res = df[label].to_numpy()
    pred = df["test_prediction"].to_numpy()
    t = (res, pred)
    return t


if __name__ == "__main__":
    t = load_test("data/input/test_pred_lab.tsv")
    print(t)
    pt = prepare_test(t, "query", "label_man")
    st = save_test(
        "intents_labelling/train_test_files/test_man.txt", pt, "query", "label_man"
    )
    model = load_mod("intents_labelling/model_files/ftext_train.bin")
    print(model_test("intents_labelling/train_test_files/test_man.txt"))
    pl = prediction_labels(model, pt, "query", "label_man")
    print(pl[0])
    print(precision_recall_fscore_support(pl[0], pl[1], average="macro"))
    cl = pt.label_man.unique().tolist()
    print(cl)
    print(classification_report(pl[0], pl[1], labels=cl))
