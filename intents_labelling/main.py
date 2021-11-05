from data_loaders import load_orcas
from intents_labelling.first_level_snorkel import SnorkelLabelling


if __name__ == "__main__":
    df = load_orcas()
    df = df[:1000]

    sl = SnorkelLabelling()
    df = sl.predict_first_level(df=df)

    outfile = "../data/output/orcas_small.tsv"
    df.to_csv(outfile, sep="\t")
