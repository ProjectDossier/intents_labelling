import csv

import pandas as pd


def load_unlabelled_orcas(data_path: str = "data/input/orcas.tsv") -> pd.DataFrame:
    """This function loads raw, unlabelled ORCAS dataset.
     It can be downloaded from: https://microsoft.github.io/msmarco/ORCAS.html"""
    names = ["qid", "query", "did", "url"]
    df = pd.read_csv(data_path, sep="\t", names=names, quoting=csv.QUOTE_NONE)
    df["query"] = df["query"].astype(str)
    df["url"] = df["url"].astype(str)

    return df


def load_labelled_orcas(
    data_path: str = "data/output/orcas_10000.tsv",
) -> pd.DataFrame:
    """This function can load any ORCAS data file that was first saved
    with the save_orcas function."""
    df = pd.read_csv(data_path, sep="\t")
    df["query"] = df["query"].astype(str)
    df["url"] = df["url"].astype(str)

    return df


def save_orcas(df, outfile: str) -> None:
    """Common function to save ORCAS file to ensure inter-operability between
    different parts of the project."""
    df.to_csv(outfile, sep="\t", index=False)
