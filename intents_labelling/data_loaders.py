import pandas as pd
import csv


def load_orcas(data_path: str = "../data/input/orcas-small.tsv") -> pd.DataFrame:
    names = ["qid", "query", "did", "url"]
    df = pd.read_csv(
        "../data/input/orcas-small.tsv", sep="\t", names=names, quoting=csv.QUOTE_NONE
    )
    return df
