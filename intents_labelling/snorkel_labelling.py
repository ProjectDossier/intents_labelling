import pandas as pd
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

from first_level_snorkel import FirstLevelIntents, first_level_functions
from second_level_snorkel import SecondLevelIntents, second_level_functions


class SnorkelLabelling:
    def __init__(self):
        self.lfs = first_level_functions
        self.second_level = second_level_functions

        self.first_level_column = "Level_1"
        self.second_level_column = "Level_2"

    def predict_first_level(self, df: pd.DataFrame) -> pd.DataFrame:
        applier = PandasLFApplier(lfs=self.lfs)
        L_train = applier.apply(df=df)

        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

        print(LFAnalysis(L=L_train, lfs=self.lfs).lf_summary())

        df.loc[:, self.first_level_column] = label_model.predict(
            L=L_train, tie_break_policy="abstain"
        )

        df.loc[
            df[self.first_level_column] == FirstLevelIntents.TRANSACTIONAL,
            self.first_level_column,
        ] = "Transactional"
        df.loc[
            df[self.first_level_column] == FirstLevelIntents.NAVIGATIONAL,
            self.first_level_column,
        ] = "Navigational"
        df.loc[
            df[self.first_level_column] == FirstLevelIntents.ABSTAIN,
            self.first_level_column,
        ] = "Abstain"

        print(df[self.first_level_column].value_counts())

        return df

    def predict_second_level(self, df: pd.DataFrame) -> pd.DataFrame:
        applier = PandasLFApplier(lfs=self.second_level)
        L_train = applier.apply(df=df)

        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

        print(LFAnalysis(L=L_train, lfs=self.second_level).lf_summary())

        df.loc[:, self.second_level_column] = label_model.predict(
            L=L_train, tie_break_policy="abstain"
        )

        df.loc[
            df[self.second_level_column] == SecondLevelIntents.FACTUAL,
            self.second_level_column,
        ] = "Factual"
        df.loc[
            df[self.second_level_column] == SecondLevelIntents.INSTRUMENTAL,
            self.second_level_column,
        ] = "Instrumental"
        df.loc[
            df[self.second_level_column] == FirstLevelIntents.ABSTAIN,
            self.second_level_column,
        ] = "Abstain"

        print(df[self.second_level_column].value_counts())

        return df

    def create_final_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a column with final label concatenating first and second level."""
        label_column = "Label"

        df[label_column] = df[self.first_level_column]
        df.loc[df[label_column] == "Abstain", label_column] = df.loc[
            df[label_column] == "Abstain", self.second_level_column
        ]
        return df
