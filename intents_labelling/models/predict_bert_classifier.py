import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from intents_labelling.data_loaders import load_labelled_orcas
import os
import json
import argparse
from intents_labelling.models.helpers import evaluate, f1_score_func, accuracy_per_class
import numpy as np


def read_labels(infile):
    with open(infile, "r") as fp:
        # json.dump(label_dict, outfile)
        return json.load(fp)

    # return labels_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert_query", type=str)
    parser.add_argument("--infile", default="data/output/orcas_10000.tsv", type=str)
    parser.add_argument("--out_path", default="models/bert/", type=str)

    args = parser.parse_args()

    # out_path = "models/bert/"
    # model_name = "bert_query"
    labels_file = "labels.json"

    if not os.path.exists(f"{args.out_path}/{args.model_name}"):
        os.makedirs(f"{args.out_path}/{args.model_name}")

    # infile = "data/output/orcas_100.tsv"

    df = load_labelled_orcas(data_path=args.infile)
    label_column = "Label"
    data_column = "query"


def predict():
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

label_dict = read_labels(infile=f"{args.out_path}/{args.model_name}/{labels_file}")
df["label"] = df[label_column].replace(label_dict)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)

# %%

model.load_state_dict(
    torch.load(
        f"{args.out_path}/{model_name}/finetuned_BERT_epoch_4.model",
        map_location=torch.device("cpu"),
    )
)

batch_size = 32

encoded_data_val = tokenizer.batch_encode_plus(
    df[data_column].values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors="pt",
)

input_ids_val = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val["attention_mask"]
labels_val = torch.tensor(df["label"].values)

dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

dataloader_validation = DataLoader(
    dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size
)

_, predictions, true_vals = evaluate(dataloader_validation, model=model, device=device)

# %%

print(accuracy_per_class(predictions, true_vals, label_dict=label_dict))
print(f1_score_func(predictions, true_vals))


# %%

model = model.to(device)

# %%

inputs = tokenizer("make America great again", return_tensors="pt").to(device)
labels = torch.tensor([0]).unsqueeze(0).to(device)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

# %%

loss

# %%

logits

# %%

labels

# %%

labels

# %%

np.exp(logits.cpu().detach().numpy()) / np.sum(np.exp(logits.cpu().detach().numpy()))

# %%

9.9927390e-01


# %%


def pred(test_data):
    l = []
    for query, label in test_data:
        inputs = tokenizer(query, return_tensors="pt").to(device)
        labels = (
            torch.tensor([label_dict[label]]).unsqueeze(0).to(device)
        )  # Batch size 1
        print(label_dict[label])
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        sm = np.exp(logits.cpu().detach().numpy()) / np.sum(
            np.exp(logits.cpu().detach().numpy())
        )
        l.append((sm, loss))
    return l


# %%

pr = pred([("medical assistance", "Factual"), ("what is my name", "Factual")])
print(pr)
print(label_dict)

# %% md
