from data_loaders import load_orcas
from intents_labelling.snorkel_labelling import SnorkelLabelling

if __name__ == "__main__":
    df = load_orcas()
    df = df.sample(1000, random_state=42)

    sl = SnorkelLabelling()
    df = sl.predict_first_level(df=df)
    df = sl.predict_second_level(df=df)

    outfile = f"../data/output/orcas_{len(df)}.tsv"
    df.to_csv(outfile, sep="\t")
