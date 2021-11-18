from data_loaders import load_unlabelled_orcas, save_orcas
from snorkel_labelling import SnorkelLabelling

if __name__ == "__main__":
    df = load_unlabelled_orcas()
    df = df.sample(10000, random_state=42)

    sl = SnorkelLabelling()
    df = sl.predict_first_level(df=df)
    df = sl.predict_second_level(df=df)

    df = sl.create_final_label(df=df)

    outfile = f"data/output/orcas_{len(df)}.tsv"
    save_orcas(df=df, outfile=outfile)
