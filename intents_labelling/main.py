from data_loaders import load_orcas


if __name__ == '__main__':
    df = load_orcas()

    outfile = "../data/output/orcas-small.tsv"
    df.to_csv(outfile, sep='\t')
