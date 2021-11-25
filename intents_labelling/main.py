from data_loaders import load_labelled_orcas, save_orcas
from snorkel_labelling import SnorkelLabelling

if __name__ == "__main__":

    df = load_labelled_orcas(data_path="data/input/orcas_train_1000000.tsv")

    sl = SnorkelLabelling()
    df = sl.predict_first_level(df=df)
    df = sl.predict_second_level(df=df)

    df = sl.create_final_label(df=df)

    train_size = 0.8
    df = sl.create_train_validation_split(df=df, train_size=train_size)

    outfile = f"data/output/orcas_{len(df)}.tsv"
    save_orcas(df=df, outfile=outfile)
