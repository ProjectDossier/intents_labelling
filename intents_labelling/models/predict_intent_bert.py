"""Short script describing how to load pretrained bert model for intent prediction on new data."""
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from intents_labelling.models.evaluation import read_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="data/test/orcas_test.tsv", type=str)
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--model_path",
        default="models/bert/finetuned_BERT_first_level.model",
        type=str,
    )
    parser.add_argument(
        "--labels_path",
        default="models/bert/labels.json",
        type=str,
    )
    args = parser.parse_args()

    label_dict = read_labels(infile=args.labels_path)
    inverse_label_dict = {v: k for k, v in label_dict.items()}

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=True)

    # load the model and update the weights
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
    )
    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=device,
        )
    )

    # load your data
    df = pd.read_csv(args.infile, sep="\t")
    INPUT_TEXT_COLUMN = "query"
    queries = df[INPUT_TEXT_COLUMN].tolist()

    # iterate over the data to get the predictions
    for query in queries:
        inputs = tokenizer(query, return_tensors="pt").to(device)
        logits = model(**inputs).logits.cpu().detach().numpy()
        predicted_intent = inverse_label_dict[np.argmax(logits)]

        print(f"{query=}, {predicted_intent=}")
