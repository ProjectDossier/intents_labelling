import string
import re
import pandas as pd


def remove_punctuation(df, data_column):
    df[data_column] = df[data_column].str.replace(
        "[" + string.punctuation + "]",
        " ",
    )
    return df


def query_plus_url(df, query_column, url_column):
    """
    combines the query and the url
    """
    df["query_url"] = "query : " + df[query_column] + " url : " + df[url_column]
    return df


def get_domains(df, url_column):
    df["domain_names"] = df[url_column].apply(
        lambda x: re.sub(
            r"(https:\/\/www\.)|(http:\/\/www\.)|(http:\/\/)|(https:\/\/)", "", str(x)
        )
    )
    df["domain_names"] = df["domain_names"].apply(
        lambda x: re.sub(r"\/.+|\/", "", str(x))
    )
    df["domain_names"] = df["domain_names"].apply(
        lambda x: re.sub(r"\.uk|\.com|\.org|\.gov|\.net", "", str(x))
    )
    return df


def get_url_stripped(df, url_column):
    df["url_strip"] = df[url_column].apply(
        lambda x: re.sub(
            r"(https:\/\/www\.)|(http:\/\/www\.)|(http:\/\/)|(https:\/\/)", "", str(x)
        )
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("data_column/input/test_raw_1000.tsv", sep="\t")

    df2 = get_domains(df, url_column="url")

