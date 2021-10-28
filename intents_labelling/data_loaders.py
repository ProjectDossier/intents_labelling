import pandas as pd

def load_orcas(data_path:str = "../data/input/orcas-small.tsv") -> pd.DataFrame:
    df = pd.read_csv(data_path, sep='\t')
    return df