import pandas as pd
import Levenshtein as lev

from intents_labelling.data_loaders import load_labelled_orcas
from intents_labelling.models.preprocessing import get_domains

def calculate_statistics(
    df: pd.DataFrame, label_column: str, url_label_mismatch: bool = False
):
    """This function calculates basic ORCAS statistics and prints them to stdout"""
    print(f"size of the dataset: {len(df)} examples")
    print(f"unique queries: {len(df['query'].unique())}")
    print(f"unique urls: {len(df['url'].unique())}")
    print(f"unique query_url pairs: {len(df[['query', 'url']].value_counts())}")
    duplicated = df[['query', 'url']].value_counts()[df[['query', 'url']].value_counts() > 1]
    print(f"duplicated query_url pairs: {len(duplicated)}")
    duplicated.to_csv("duplicated.csv")

    df = get_domains(df=df, url_column="url")
    print(f"unique domains: {len(df['domain_names'].unique())}")

    df['query_len'] = df['query'].astype(str).str.split().str.len()
    print(f"Mean query length: {df['query_len'].mean()}")

    print("items per label:")
    print(df[label_column].value_counts())

    print("\n")
    print("label distribution:")
    print(100 * df[label_column].value_counts() / len(df))

    lev_list = []
    for url, query in zip(df["url"].tolist(), df["query"].tolist()):
        lev_list.append(lev.distance(str(url), str(query)))
    df["levenshtein"] = lev_list
    print("\n")
    print("Mean Levenshtein distance between query and url:")
    print(df.groupby(label_column)["levenshtein"].mean())

    if url_label_mismatch:
        # TODO: finish this part
        print("\n")
        print("queries with the same url but different labels:")
        for url in (
            df["url"].value_counts()[df["url"].value_counts() > 1].index.tolist()
        ):
            if len(df.loc[df["url"] == url, label_column].unique()) > 1:
                print(df.loc[df["url"] == url])


if __name__ == "__main__":
    dataset_path = "data/output/orcas_2000000.tsv"
    df = load_labelled_orcas(dataset_path)
    calculate_statistics(df=df, label_column="Label")
