import pandas as pd
import Levenshtein as lev


def calculate_statistics(
    df: pd.DataFrame, label_column: str, url_label_mismatch: bool = False
):
    print(f"size of the dataset: {len(df)} examples")
    print(f"unique queries: {len(df['query'].unique())}")
    print(f"unique urls: {len(df['url'].unique())}")
    print(f"unique query_url pairs: {len(df[['query', 'url']].value_counts())}")

    print("\n")
    print("label distribution:")
    print(100 * df[label_column].value_counts() / len(df))

    lev_list = []
    for url, query in zip(df["url"].tolist(), df["query"].tolist()):
        lev_list.append(lev.distance(url, query))
    df["levenshtein"] = lev_list
    print("\n")
    print("Mean Levenshtein distance between query and url:")
    print(df.groupby(label_column)["levenshtein"].mean())

    if url_label_mismatch:
        print("\n")
        print("queries with the same url but different labels:")
        for url in (
            df["url"].value_counts()[df["url"].value_counts() > 1].index.tolist()
        ):
            if len(df.loc[df["url"] == url, label_column].unique()) > 1:
                print(df.loc[df["url"] == url])


if __name__ == "__main__":
    dataset_path = "data/test/orcas_test.tsv"
    df = pd.read_csv(dataset_path, sep="\t")
    calculate_statistics(df=df, label_column="label_manual")
