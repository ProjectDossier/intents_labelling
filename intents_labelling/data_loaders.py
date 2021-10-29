import pandas as pd
import csv


def load_orcas(data_path: str = "../data/input/orcas_small.tsv") -> pd.DataFrame:
    names = ["qid", "query", "did", "url"]
    df = pd.read_csv(
        data_path, sep="\t", names=names, quoting=csv.QUOTE_NONE
    )
    df["query"] = df["query"].astype(str)

    return df
