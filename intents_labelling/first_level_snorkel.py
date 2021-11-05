import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="query", doc_field="doc", memoize=True)


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


@labeling_function()
def lf_download_lookup(x):
    keywords = ["download", "obtain", "access", "earn", "redeem"]
    return (
        TRANSACTIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


@labeling_function()
def lf_audio_video_lookup(x):
    keywords = ["audio", "video", "image", "images"]
    return (
        TRANSACTIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


movies_df = pd.read_csv("../data/helpers/movies.csv")
movie_names_list = movies_df["title"].str.lower().tolist()


@labeling_function(pre=[spacy])
def lf_movie_name_lookup(x):
    if x.doc[0].text.lower() in informational_start_words:
        return ABSTAIN
    else:
        return (
            TRANSACTIONAL
            if any(movie_name in x.query.lower() for movie_name in movie_names_list)
            else ABSTAIN
        )


with open("../data/helpers/common_extensions.txt") as fp:
    common_extensions_list = [line.strip() for line in fp.readlines()]


@labeling_function()
def lf_extension_lookup(x):
    return (
        TRANSACTIONAL
        if any(word in x.query.lower().split() for word in common_extensions_list)
        else ABSTAIN
    )


@labeling_function()
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
    return (
        TRANSACTIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


"""NAVIGATIONAL Labelling functions"""


@labeling_function()
def lf_www_lookup(x):
    keywords = ["www", "http", "https"]
    return (
        NAVIGATIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


with open("../data/helpers/top_level_domains.txt") as fp:
    domain_names_list = [line.strip() for line in fp.readlines()]


@labeling_function()
def lf_domain_name_lookup(x):
    return (
        NAVIGATIONAL
        if any(word in x.query.lower() for word in domain_names_list)
        else ABSTAIN
    )


@labeling_function()
def lf_login_lookup(x):
    keywords = ["login", "signin", "log in", "sign in", "signup", "sign up"]
    return (
        NAVIGATIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


@labeling_function(pre=[spacy])
def lf_has_ner(x):
    for ent in x.doc.ents:
        if (
            ent.label_ in ["ORG", "PERSON"]
            and x.doc[0].text.lower() not in informational_start_words
        ):
            return NAVIGATIONAL
    else:
        return ABSTAIN


class SnorkelLabelling:
    def __init__(self):
        self.lfs = [
            lf_download_lookup,
            lf_audio_video_lookup,
            lf_movie_name_lookup,
            lf_extension_lookup,
            lf_transaction_lookup,
            lf_www_lookup,
            lf_domain_name_lookup,
            lf_login_lookup,
            lf_has_ner,
        ]

    def predict_first_level(self, df: pd.DataFrame) -> pd.DataFrame:
        applier = PandasLFApplier(lfs=self.lfs)
        L_train = applier.apply(df=df)

        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

        print(LFAnalysis(L=L_train, lfs=self.lfs).lf_summary())

        df.loc[:, "Labels"] = label_model.predict(L=L_train, tie_break_policy="abstain")

        df.loc[df["Labels"] == TRANSACTIONAL, "Labels"] = "Transactional"
        df.loc[df["Labels"] == NAVIGATIONAL, "Labels"] = "Navigational"
        df.loc[df["Labels"] == ABSTAIN, "Labels"] = "Abstain"

        print(df["Labels"].value_counts())

        return df