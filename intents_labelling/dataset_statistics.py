"""This scripts calculcates dataset statistics used in the paper for
labelled ORCAS datasets."""
import argparse

import pandas as pd
import Levenshtein as lev

from intents_labelling.data_loaders import load_labelled_orcas
from intents_labelling.models.preprocessing import get_domains


def calculate_statistics(
    df: pd.DataFrame,
    label_column: str,
    url_label_mismatch: bool = False,
    query_label_mismatch: bool = False,
):
    """This function calculates basic ORCAS statistics and prints them to stdout"""
    print(f"size of the dataset: {len(df)} examples")
    print(f"unique queries: {len(df['query'].unique())}")
    print(f"unique urls: {len(df['url'].unique())}")
    print(f"unique query_url pairs: {len(df[['query', 'url']].value_counts())}")
    duplicated = df[["query", "url"]].value_counts()[
        df[["query", "url"]].value_counts() > 1
    ]
    print(f"duplicated query_url pairs: {len(duplicated)}")
    duplicated.to_csv("duplicated.csv")

    df = get_domains(df=df, url_column="url")
    print(f"unique domains: {len(df['domain_names'].unique())}")

    df["query_len"] = df["query"].astype(str).str.split().str.len()
    print(f"Mean query length: {df['query_len'].mean()}")

    print("Mean query length by category:")
    print(df.groupby(label_column)["query_len"].mean())

    for length in [1, 2, 3, 4]:
        print(f"Count of queries of length == {length}:")
        print(df[df["query_len"] == length].groupby(label_column)["query"].count())
        print(f"total len for len={length} == {len(df[df['query_len'] == length])}")

    length = 5
    print(f"Count of queries of length >= {length}:")
    print(df[df["query_len"] >= length].groupby(label_column)["query"].count())
    print(f"total len for len>={length} == {len(df[df['query_len'] >= length])}")

    print("\n")
    unique_words = []
    for x in df["query"].str.split(" ").tolist():
        unique_words.extend(x)
        unique_words = list(set(unique_words))
    print(f"Unique words: {len(unique_words)}")
    print("\n\n\n")

    print("items per label:")
    print(df[label_column].value_counts())

    print("\n")
    print("label distribution:")
    print(100 * df[label_column].value_counts() / len(df))

    # mean levenshtein distance between query and url
    lev_list = []
    for url, query in zip(df["url"].tolist(), df["query"].tolist()):
        lev_list.append(lev.distance(str(url), str(query)))
    df["levenshtein"] = lev_list
    print("\n")
    print(f'Mean Levenshtein distance: {df["levenshtein"].mean()}')
    print("Mean Levenshtein distance between query and url:")
    print(df.groupby(label_column)["levenshtein"].mean())

    if url_label_mismatch:
        print("\n")
        print("different queries with the same url but different labels:")
        for url in (
            df["url"].value_counts()[df["url"].value_counts() > 5].index.tolist()
        ):
            if len(df.loc[df["url"] == url, label_column].unique()) > 1:
                print(df.loc[df["url"] == url][["query", "url"]])
                print(df.loc[df["url"] == url].groupby(label_column)["url"].count())

    if query_label_mismatch:
        print("\n")
        print("different queries with the same url but different labels:")
        for url in (
            df["query"].value_counts()[df["query"].value_counts() > 10].index.tolist()
        ):
            if len(df.loc[df["query"] == url, label_column].unique()) > 2:
                print(
                    df.loc[df["query"] == url][
                        ["query", "url", label_column]
                    ].sort_values([label_column, "url"])
                )
                print(df.loc[df["query"] == url].groupby(label_column)["query"].count())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data/output/orcas_2000000.tsv")
    parser.add_argument(
        "--label_column",
        default="Label",
        help="'Label' for Snorkel output, 'label_manual' for manual annotations",
    )
    args = parser.parse_args()

    df = load_labelled_orcas(args.dataset_path)
    calculate_statistics(
        df=df,
        label_column=args.label_column,
        url_label_mismatch=False,
        query_label_mismatch=False,
    )
