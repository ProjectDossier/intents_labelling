import re
from enum import IntEnum

import pandas as pd
from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="query", doc_field="doc", memoize=True)


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

"""TRANSACTIONAL Labelling functions"""


@labeling_function(pre=[spacy])
def lf_download_lookup(x):
    transactional_keywords = [
        "download",
        "obtain",
        "access",
        "earn",
        "redeem",
        "watch",
        "install",
        "play",
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
                for word in transactional_keywords
            )
            else FirstLevelIntents.ABSTAIN
        )


@labeling_function(pre=[spacy])
def lf_audio_video_lookup(x):
    if x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        keywords = ["audio", "video", "image", "images"]
        return (
            FirstLevelIntents.TRANSACTIONAL
            if any(word in x.query.lower() for word in keywords)
            else FirstLevelIntents.ABSTAIN
        )


@labeling_function(pre=[spacy])
def lf_transaction_lookup(x):
    keywords = [
        "online",
        "free",
        "transaction",
        "buy",
        "chat",
        "purchase",
        "shop for",
        "procure",
        "complimentary",
        "gratuitous",
        "payment",
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
    if any(word in x.query.lower() for word in transactional_keywords):
        return FirstLevelIntents.TRANSACTIONAL
    elif any(word in x.query.lower() for word in factual_keywords):
        return FirstLevelIntents.ABSTAIN
    elif x.doc[0].text.lower() in informational_start_words:
        return FirstLevelIntents.ABSTAIN
    else:
        li = list(x.query.lower().split(" "))
        r1 = re.search(r"https:\/\/www\.(.*?)\/", x.url)
        r2 = re.search(r"http:\/\/www\.(.*?)\/", x.url)
        r3 = re.search(r"http:(.*?)\/", x.url)
        st = ""
        if r1:
            st = r1.group(1)
        elif r2:
            st = r2.group(1)
        elif r3:
            st = r3.group(1)
        return (
            FirstLevelIntents.NAVIGATIONAL
            if any(word in st.lower() for word in li)
            else FirstLevelIntents.ABSTAIN
        )


first_level_functions = [
    lf_match_url,
    lf_download_lookup,
    lf_audio_video_lookup,
    lf_transaction_lookup,
    lf_www_lookup,
    lf_domain_name_lookup,
    lf_login_lookup,
]
