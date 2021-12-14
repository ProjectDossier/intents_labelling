from intents_labelling.models.helpers import (
    precision_score_func,
    recall_score_func,
    f1_score_func,
)
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def test_load(
    data_path: str = "",
) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="\t")
    return df


def get_labels(df):
    df["query"] = df["query"].astype(str)
    df["url"] = df["url"].astype(str)
    lab_pred = df["Label"].tolist()
    lab_true = df["label_man"].tolist()
    t = (lab_true, lab_pred)
    return t


if __name__ == "__main__":
    df = test_load("data/input/test_pred_lab.tsv")
    t = get_labels(df)
    p = precision_score(t[0], t[1], average="macro")
    r = recall_score(t[0], t[1], average="macro")
    f1 = f1_score(t[0], t[1], average="macro")
    cl = df.label_man.unique().tolist()
    print("cl_labels", cl)
    print("Precision score: ", p)
    print("Recall score: ", r)
    print("F1 score: ", f1)
    print(classification_report(t[0], t[1], labels=cl))
