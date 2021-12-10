import re
from enum import IntEnum

from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(
    text_field="query", doc_field="doc", language="en_core_web_lg", memoize=True
)


class SecondLevelIntents(IntEnum):
    INSTRUMENTAL = 1
    FACTUAL = 0
    ABSTAIN = -1


transactional_verbs = [
    "download",
    "downloading" "obtain",
    "obtaining" "access",
    "accessing" "watch",
    "watching",
    "install",
    "installing",
    "play",
    "playing",
    "listen",
    "listening",
    "buy",
    "buying",
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

factual_keywords = [
    "phone",
    "code",
    "number",
    "zip",
    "facts",
    "statistics",
    "quantity",
    "quantities",
    "average",
    "sum",
    "cost",
    "amount",
    "pay",
]
"""INSTRUMENTAL Labelling functions"""

with open("data/helpers/verbs.txt") as fp:
    verb_list = [line.strip() for line in fp.readlines()]


@labeling_function(pre=[spacy])
def lf_is_verb(x):
    if any(re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I) for word in factual_keywords):
        return SecondLevelIntents.FACTUAL
    elif x.doc[0].text in verb_list:
        return SecondLevelIntents.INSTRUMENTAL
    else:
        return SecondLevelIntents.ABSTAIN


@labeling_function(pre=[spacy])
def lf_is_ing_verb(x):
    if x.doc[0].text in transactional_verbs:
        return SecondLevelIntents.ABSTAIN
    elif x.doc[0].pos_ == "VERB" and "ing" in x.doc[0].text:
        return SecondLevelIntents.INSTRUMENTAL
    else:
        return SecondLevelIntents.ABSTAIN


@labeling_function()
def lf_howto(x):
    keywords = ["how to", "how do", "how can", "how does"]
    return (
        SecondLevelIntents.INSTRUMENTAL
        if any(
            re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
            for word in keywords
        )
        else SecondLevelIntents.ABSTAIN
    )

@labeling_function()
def lf_wikihow_lookup(x):
    urls = [
        "www.wikihow.com",
        "support.apple.com",
        "support.office.com",
        "support.google.com",
    ]
    return (
        SecondLevelIntents.INSTRUMENTAL
        if any(url in x.url.lower() for url in urls)
        else SecondLevelIntents.ABSTAIN
    )


"""FACTUAL Labelling functions"""


@labeling_function()
def lf_keyword_lookup(x):
    keywords = ["why", "what", "when", "who", "where", "how"]
    how_words = ["how to", "how do", "how does", "how can"]
    return (
        SecondLevelIntents.FACTUAL
        if any(word in x.query.lower() for word in keywords)
        and not any(word in x.query.lower() for word in how_words)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_question_words(x):
    keywords = ["is", "can", "do", "does", "did"]
    return (
        SecondLevelIntents.FACTUAL
        if any(x.query.lower().startswith(word) for word in keywords)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_facts_lookup(x):
    keywords = [
        "facts",
        "statistics",
        "quantity",
        "quantities",
        "recipe",
        "side effects",
    ]
    return (
        SecondLevelIntents.FACTUAL
        if any(
            re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
            for word in keywords
        )
        else SecondLevelIntents.ABSTAIN
    )

@labeling_function()
def lf_finance_lookup(x):
    keywords = [
        "average",
        "sum",
        "cost",
        "costs",
        "amount",
        "amounts",
        "salary",
        "salaries",
        "pay",
    ]
    return (
        SecondLevelIntents.FACTUAL
        if any(
            re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
            for word in keywords
        )
        else SecondLevelIntents.ABSTAIN
    )

@labeling_function()
def lf_phone(x):
    keywords = [
        "number",
        "numbers",
        "phone",
        "phones",
        "code",
        "codes",
        "zip",
    ]
    return (
        SecondLevelIntents.FACTUAL
        if any(
            re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
            for word in keywords
        )
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_definition(x):
    keywords = [
        "define",
        "definition",
        "definitions",
        "meaning",
        "meanings",
        "example",
        "examples",
        "side effect",
        "side effects",
    ]
    return (
        SecondLevelIntents.FACTUAL
        if any(
            re.search(rf"(?:\s|^){word}(?:\s|$)", x.query, flags=re.I)
            for word in keywords
        )
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_digit(x):
    return (
        SecondLevelIntents.FACTUAL
        if re.search(r"\d", x.query, flags=re.I)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_url_lookup(x):
    urls = [
        "www.accuweather.com",
        "weather.com",
        "www.goodreads.com",
        "merriam-webster.com",
        "drugs.com",
        "imdb.com",
        "dictionary.com",
        "reference.com",
        "britannica.com",
        "timeanddate.com",
        "allrecipes.com",
        "thesaurus.com",
        "thefreedictionary.com",
        "thoughtco.com",
        "healthline.com",
        "mayoclinic.org",
        "medicinenet.com",
        "webmd.com",
    ]
    return (
        SecondLevelIntents.FACTUAL
        if any(url in x.url.lower() for url in urls)
        else SecondLevelIntents.ABSTAIN
    )



@labeling_function(pre=[spacy])
def lf_wiki(x):
    if "wikipedia.org" in x.url.lower():
            return SecondLevelIntents.FACTUAL
    else:
        return SecondLevelIntents.ABSTAIN


second_level_functions = [
    lf_is_verb,
    lf_howto,
    lf_is_ing_verb,
    lf_wikihow_lookup,
    lf_keyword_lookup,
    lf_question_words,
    lf_facts_lookup,
    lf_finance_lookup,
    lf_phone,
    lf_definition,
    lf_digit,
    lf_url_lookup,
    lf_wiki
]
