import re
from enum import IntEnum
import Levenshtein as lev

import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(
    text_field="query", doc_field="doc", language="en_core_web_lg", memoize=True
)


class FirstLevelIntents(IntEnum):
    TRANSACTIONAL = 1
    NAVIGATIONAL = 0
    ABSTAIN = -1


informational_start_words = [
    "why",
    "what",
    "when",
    "who",
    "where",
    "how",
    "is",
    "can",
    "do",
    "does",
    "did",
]

factual_keywords = [
    "define",
    "definition",
    "meaning",
    "phone",
    "code",
    "number",
    "zip",
    "facts",
    "statistics",
    "quantity",
    "quantities",
    "recipe",
    "recipes",
    "side effects",
    "weather",
    "average",
    "sum",
    "cost",
    "amount",
    "salary",
    "salaries",
    "pay",
]
transactional_keywords = [
    "download",
    "obtain",
    "access",
    "earn",
    "redeem",
    "watch",
    "install",
    "app",
    "application",
    "play",
    "listen",
    "online",
    "free",
    "buy",
    "payment",
    "audio",
    "video",
    "image",
]

ne_labels = [
    "ORG",
    "PERSON",
    "EVENT",
    "FAC",
    "LOC",
    "GPE",
    "PRODUCT",
    "WORK_OF_ART",
]


"""TRANSACTIONAL Labelling functions"""


@labeling_function(pre=[spacy])
def lf_download_lookup(x):
    transact_keywords = [
        "download",
        "obtain",
        "access",
        "earn",
        "redeem",
        "watch",
        "install",
        "play",
        "app",
        "listen",
        "loader",
    ]
    if x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        return (
            FirstLevelIntents.TRANSACTIONAL
            if any(
                re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
                for word in transact_keywords
            )
            else FirstLevelIntents.ABSTAIN
        )


@labeling_function(pre=[spacy])
def lf_audio_video_lookup(x):
    keywords = [
        "audio",
        "video",
        "videos",
        "image",
        "images",
        "calculator"
    ]
    if x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        return (
            FirstLevelIntents.TRANSACTIONAL
            if any(
                re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
                for word in keywords
            )
            else FirstLevelIntents.ABSTAIN
        )


@labeling_function(pre=[spacy])
def lf_transaction_lookup(x):
    keywords = [
        "online",
        "free",
        "buy",
        "chat",
        "purchase",
        "shop for",
        "procure",
        "complimentary",
        "gratuitous",
        "payment",
        "converter",
        "convertor",
        "converters",
        "convertors"
        "viewer",
        "crop",
    ]
    if x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        return (
            FirstLevelIntents.TRANSACTIONAL
            if any(
                re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
                for word in keywords
            )
            else FirstLevelIntents.ABSTAIN
        )


"""NAVIGATIONAL Labelling functions"""


@labeling_function(pre=[spacy])
def lf_www_lookup(x):
    keywords = ["www", "http", "https"]
    if x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        return (
            FirstLevelIntents.NAVIGATIONAL
            if any(word in x.query.lower() for word in keywords)
            else FirstLevelIntents.ABSTAIN
        )


with open("data/helpers/top_level_domains.txt") as fp:
    domain_names_list = [line.strip() for line in fp.readlines()]


@labeling_function(pre=[spacy])
def lf_domain_name_lookup(x):
    if x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        return (
            FirstLevelIntents.NAVIGATIONAL
            if any(word in x.query.lower() for word in domain_names_list)
            else FirstLevelIntents.ABSTAIN
        )


@labeling_function(pre=[spacy])
def lf_login_lookup(x):
    keywords = [
        "login",
        "signin",
        "log in",
        "sign in",
        "signup",
        "sign up",
        "site",
        "account",
        "website",
    ]
    if x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        return (
            FirstLevelIntents.NAVIGATIONAL
            if any(
                re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
                for word in keywords
            )
            else FirstLevelIntents.ABSTAIN
        )


@labeling_function(pre=[spacy])
def lf_match_url(x):
    if any(re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
           for word in transactional_keywords):
        return FirstLevelIntents.TRANSACTIONAL
    elif any(re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
             for word in factual_keywords):
        return FirstLevelIntents.ABSTAIN
    elif x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        r = re.search(r"https:\/\/www\.(.*?)\/|http:\/\/www\.(.*?)\/|http:\/\/(.*?)\/|https:\/\/(.*?)\/", x.url)
        st = ""
        for i in range(1,5):
            if r.group(i) is not None:
             st = r.group(i)
        st = re.sub(r"\.uk|\.com|\.org|\.gov|\.net", "", st)
        st_q = x.query.lower()
        st_url = st.lower()
        ratio = lev.ratio(st_q,st_url)
        if (ratio >= 0.55):
            return FirstLevelIntents.NAVIGATIONAL
        else:
            return FirstLevelIntents.ABSTAIN


first_level_functions = [
    lf_match_url,
    lf_download_lookup,
    lf_audio_video_lookup,
    lf_transaction_lookup,
    lf_www_lookup,
    lf_domain_name_lookup,
    lf_login_lookup,
]
