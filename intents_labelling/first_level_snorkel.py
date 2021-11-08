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
]

"""TRANSACTIONAL Labelling functions"""

# first word == watch to transactional


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


# movies_df = pd.read_csv("../data/helpers/movies.csv")
# movie_names_list = movies_df["title"].str.lower().tolist()
#
#
# @labeling_function(pre=[spacy])
# def lf_movie_name_lookup(x):
#     if x.doc[0].text.lower() in informational_start_words:
#         return FirstLevelIntents.ABSTAIN
#     else:
#         return (
#             FirstLevelIntents.TRANSACTIONAL
#             if any(
#                 movie_name.strip() == x.query.lower().strip()
#                 for movie_name in movie_names_list
#             )
#             else FirstLevelIntents.ABSTAIN
#         )
#
#     # url


# with open("../data/helpers/common_extensions.txt") as fp:
#     common_extensions_list = [line.strip() for line in fp.readlines()]
#
#
# @labeling_function(pre=[spacy])
# def lf_extension_lookup(x):
#     if x.doc[0].text.lower() in informational_start_words:
#         return FirstLevelIntents.ABSTAIN
#     else:
#         return (
#             FirstLevelIntents.TRANSACTIONAL
#             if any(
#                 re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
#                 for word in common_extensions_list
#             )
#             else FirstLevelIntents.ABSTAIN
#         )


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


with open("../data/helpers/top_level_domains.txt") as fp:
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
    keywords = ["login", "signin", "log in", "sign in", "signup", "sign up"]
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


# # to be removed? if url contains wikipedia then it is factual?
# @labeling_function(pre=[spacy])
# def lf_has_ner(x):
#     if x.doc[0].text.lower() in informational_start_words:
#         return FirstLevelIntents.ABSTAIN
#     else:
#         for ent in x.doc.ents:
#             if (
#                 ent.label_ in ["ORG", "PERSON"]
#                 and x.doc[0].text.lower() not in informational_start_words
#             ):
#                 return FirstLevelIntents.NAVIGATIONAL
#         else:
#             return FirstLevelIntents.ABSTAIN
#

first_level_functions = [
    lf_download_lookup,
    lf_audio_video_lookup,
    # lf_movie_name_lookup,
    # lf_extension_lookup,
    lf_transaction_lookup,
    lf_www_lookup,
    lf_domain_name_lookup,
    lf_login_lookup,
    # lf_has_ner,
]
