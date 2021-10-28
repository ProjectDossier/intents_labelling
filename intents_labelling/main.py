from data_loaders import load_orcas
from intents_labelling.labelling_snorkel import SnorkelLabelling


if __name__ == "__main__":
    df = load_orcas()

    sl = SnorkelLabelling()
    df = sl.predict_transactional(df=df)

    outfile = "../data/output/orcas-small.tsv"
    df.to_csv(outfile, sep="\t")
