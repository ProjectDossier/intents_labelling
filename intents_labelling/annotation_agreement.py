from sklearn.metrics import cohen_kappa_score
import pandas as pd

if __name__ == "__main__":
    df1 = pd.read_csv("data/test/annotator1.tsv", sep="\t")
    df2 = pd.read_csv("data/test/annotator2.tsv", sep="\t")

    y1 = df1["label"].tolist()
    y2 = df2["label"].tolist()

    kappa = cohen_kappa_score(y1=y1, y2=y2)
    print(f"{kappa=}")
