import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

"""
Evaluation of Snorkel results against the test set
"""


def data_load(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="\t")
    return df


def merge_res_pred(df_res, df_test):
    label_manual = df_test["label_manual"]
    df_test_pred = df_res.join(label_manual)
    return df_test_pred


def get_labels(df_test_pred):
    df_test_pred["query"] = df_test_pred["query"].astype(str)
    df_test_pred["url"] = df_test_pred["url"].astype(str)
    lab_true = df_test_pred["label_manual"].tolist()
    lab_pred = df_test_pred["Label"].tolist()
    t = (lab_true, lab_pred)
    return t


if __name__ == "__main__":
    df_test = data_load("data/test/orcas_test.tsv")
    df_pred = data_load("data/output/orcas_res.tsv")
    df_test_pred = merge_res_pred(df_pred, df_test)
    t = get_labels(df_test_pred)
    p = precision_score(t[0], t[1], average="macro")
    r = recall_score(t[0], t[1], average="macro")
    f1 = f1_score(t[0], t[1], average="macro")
    cl = df_test_pred.label_manual.unique().tolist()
    print("cl_labels", cl)
    print("Precision score: ", p)
    print("Recall score: ", r)
    print("F1 score: ", f1)
    print(classification_report(t[0], t[1], labels=cl))
