import string
import re
import pandas as pd



def remove_punctuation(df,data):
    df[data] = df[data].str.replace('[' + string.punctuation + ']', ' ', )

def query_plus_url(df, query, url):
    """
    combines the query and the url
    """
    df["query_url"] = "query : " + df[query] + " url : " + df[url]
    return df

def get_domains(df,url):
    df["domain_names"] = df[url].apply(lambda x: re.sub(r'(https:\/\/www\.)|(http:\/\/www\.)|(http:\/\/)|(https:\/\/)', '', str(x)))
    df["domain_names"] = df["domain_names"].apply(lambda x: re.sub(r"\/.+|\/", '', str(x)))
    df["domain_names"] = df["domain_names"].apply(lambda x: re.sub(r"\.uk|\.com|\.org|\.gov|\.net", '', str(x)))
    return df
"""
def get_domains(df,url):
    l = df[url].to_list()
    l_stripped = []
    df["domain_names"] = df[url].str.replace(r"(https:\/\/www\.)|(http:\/\/www\.)|(http:\/\/)|(https:\/\/)","")
    df["domain_names"] = df["domain_names"].str.replace(r"\/.+|\/", "")
    df["domain_names"] = df["domain_names"].str.replace(r"\.uk|\.com|\.org|\.gov|\.net", "")
    return df
"""

def get_url_stripped(df,url):
    df["url_strip"] = df[url].apply(lambda x: re.sub(r'(https:\/\/www\.)|(http:\/\/www\.)|(http:\/\/)|(https:\/\/)', '', str(x)))
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/input/test_raw_1000.tsv", sep="\t")
    df.head()
    df2 = get_domains(df,"url")
    df2.head()
