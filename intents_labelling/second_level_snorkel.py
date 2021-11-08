import re
from enum import IntEnum

from snorkel.labeling import labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="query", doc_field="doc", memoize=True)


class SecondLevelIntents(IntEnum):
    INSTRUMENTAL = 1
    FACTUAL = 0
    ABSTAIN = -1


"""INSTRUMENTAL Labelling functions"""


@labeling_function(pre=[spacy])
def lf_is_verb(x):
    if x.doc[0].pos_ == "VERB" and x.doc[0].text == x.doc[0].lemma_:
        return SecondLevelIntents.INSTRUMENTAL
    else:
        return SecondLevelIntents.ABSTAIN


@labeling_function()
def lf_howto(x):
    keywords = ["how to"]
    return (
        SecondLevelIntents.INSTRUMENTAL
        if any(word in x.query.lower() for word in keywords)
        else SecondLevelIntents.ABSTAIN
    )


"""FACTUAL Labelling functions"""


@labeling_function()
def lf_keyword_lookup(x):
    keywords = ["why", "what", "when", "who", "where", "how"]
    return (
        SecondLevelIntents.FACTUAL
        if any(
            word in x.query.lower() and "how to" not in x.query.lower()
            for word in keywords
        )
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_question_words(x):
    keywords = ["is", "can", "do", "does"]
    return (
        SecondLevelIntents.FACTUAL
        if any(x.query.lower().startswith(word) for word in keywords)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_facts_lookup(x):
    keywords = ["facts", "statistics", "quantity", "quantities"]
    return (
        SecondLevelIntents.FACTUAL
        if any(word in x.query.lower() for word in keywords)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_finance_lookup(x):
    keywords = ["average", "sum", "cost", "amount", "salary", "salaries", "pay"]
    return (
        SecondLevelIntents.FACTUAL
        if any(word in x.query.lower() for word in keywords)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_phone(x):
    keywords = ["number", "phone", "code", "zip"]
    return (
        SecondLevelIntents.FACTUAL
        if any(word in x.query.lower() for word in keywords)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_definition(x):
    keywords = ["define", "definition", "meaning"]
    return (
        SecondLevelIntents.FACTUAL
        if any(word in x.query.lower() for word in keywords)
        else SecondLevelIntents.ABSTAIN
    )


@labeling_function()
def lf_digit(x):
    return (
        SecondLevelIntents.FACTUAL
        if re.search(r"\d", x.query, flags=re.I)
        else SecondLevelIntents.ABSTAIN
    )


second_level_functions = [
    lf_is_verb,
    lf_howto,
    lf_keyword_lookup,
    lf_question_words,
    lf_facts_lookup,
    lf_finance_lookup,
    lf_phone,
    lf_definition,
    lf_digit,
]
