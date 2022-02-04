import pandas as pd

from intents_labelling.data_loaders import (
    load_labelled_orcas,
    load_unlabelled_orcas,
    save_orcas,
)


def prepare_train_set(
    orcas_df: pd.DataFrame, test_df: pd.DataFrame, train_size: int
) -> pd.DataFrame:
    """Function prepares sample of the training data.
    It makes sure that none of the queries selected for the training data sample
    does not appear in the test dataset."""

    test_df["test"] = True
    merged_df = pd.merge(orcas_df, test_df, how="outer")
    df = orcas_df[orcas_df.index.isin(merged_df[merged_df["test"] != True].index)]

    df = df.sample(train_size, random_state=42)

    return df


if __name__ == "__main__":
    testset_location = "data/test/orcas_test.tsv"
    raw_orcas_location = "data/input/orcas.tsv"

    train_size = 2000000
    trainset_location = f"data/input/orcas_train_{train_size}.tsv"

    test_df = load_labelled_orcas(data_path=testset_location)
    orcas_df = load_unlabelled_orcas(data_path=raw_orcas_location)

    train_df = prepare_train_set(
        orcas_df=orcas_df, test_df=test_df, train_size=train_size
    )

    save_orcas(df=train_df, outfile=trainset_location)
