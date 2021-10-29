import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel


TRANSACTIONAL = 1
NAVIGATIONAL = 0
ABSTAIN = -1


"""TRANSACTIONAL Labelling functions"""


@labeling_function()
def lf_keyword_lookup(x):
    keywords = ["why", "what", "when", "who", "where", "how"]
    return (
        TRANSACTIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


@labeling_function()
def lf_download_lookup(x):
    keywords = ["download", "obtain"]
    return (
        TRANSACTIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


@labeling_function()
def lf_audio_video_lookup(x):
    keywords = ["audio", "video", "image", "images"]
    return (
        TRANSACTIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


@labeling_function()
def lf_extension_lookup(x):
    keywords = ["jpeg", "zip", "rar", "png", "mp3"]
    return (
        TRANSACTIONAL if any(word in x.query.lower() for word in keywords) else ABSTAIN
    )


@labeling_function()
def lf_transaction_lookup(x):
    keywords = ["online", "free", "transaction", "buy"]
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


class SnorkelLabelling:
    def __init__(self):
        self.lfs = [
            lf_keyword_lookup,
            lf_download_lookup,
            lf_audio_video_lookup,
            lf_extension_lookup,
            lf_transaction_lookup,
            lf_www_lookup,
            lf_domain_name_lookup,
        ]

    def predict_first_level(self, df: pd.DataFrame) -> pd.DataFrame:
        applier = PandasLFApplier(lfs=self.lfs)
        L_train = applier.apply(df=df)

        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

        print(LFAnalysis(L=L_train, lfs=self.lfs).lf_summary())

        df["Labels"] = label_model.predict(L=L_train, tie_break_policy="abstain")

        df.loc[df["Labels"] == TRANSACTIONAL, "Labels"] = "Transactional"
        df.loc[df["Labels"] == NAVIGATIONAL, "Labels"] = "Navigational"
        df.loc[df["Labels"] == ABSTAIN, "Labels"] = "Abstain"

        print(df["Labels"].value_counts())

        return df
