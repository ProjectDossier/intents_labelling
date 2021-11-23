import argparse
import json
import os.path
import random
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from intents_labelling.data_loaders import load_labelled_orcas
from intents_labelling.models.helpers import (
    f1_score_func,
    recall_score_func,
    precision_score_func,
    evaluate,
)

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(0)


def get_label_dict(labels: List[str]) -> Dict[str, int]:
    label_dict = {}
    for index, possible_label in enumerate(labels):
        label_dict[possible_label] = index
    return label_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert_query", type=str)
    parser.add_argument("--infile", default="data/output/orcas_10000.tsv", type=str)
    parser.add_argument("--out_path", default="models/bert/", type=str)

    args = parser.parse_args()

    labels_file = "labels.json"

    if not os.path.exists(f"{args.out_path}/{args.model_name}"):
        os.makedirs(f"{args.out_path}/{args.model_name}")

    label_column = "Label"
    data_column = "query"

    df = load_labelled_orcas(data_path=args.infile)

    label_dict = get_label_dict(df[label_column].unique().tolist())

    with open(f"{args.out_path}/{args.model_name}/{labels_file}", "w") as outfile:
        json.dump(label_dict, outfile)

    df["label"] = df[label_column].replace(label_dict)

    print(df["data_type"].value_counts())

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    df[data_column] = df[data_column].replace(np.nan, "0")

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == "train"][data_column].values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == "validation"][data_column].values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )

    input_ids_train = encoded_data_train["input_ids"]
    print(input_ids_train)
    attention_masks_train = encoded_data_train["attention_mask"]
    print(attention_masks_train)
    labels_train = torch.tensor(df[df.data_type == "train"].label.values)

    input_ids_val = encoded_data_val["input_ids"]
    attention_masks_val = encoded_data_val["attention_mask"]
    labels_val = torch.tensor(df[df.data_type == "validation"].label.values)

    # %%

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # %%

    len(dataset_train), len(dataset_val)

    # %%

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
    )

    # %%

    batch_size = 32

    dataloader_train = DataLoader(
        dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size
    )

    dataloader_validation = DataLoader(
        dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size
    )

    # %%

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

    # %%

    epochs = 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs
    )

    torch.cuda.empty_cache()
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # %%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(device)

    for epoch in tqdm(range(1, epochs + 1)):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(
            dataloader_train,
            desc="Epoch {:1d}".format(epoch),
            leave=False,
            disable=False,
        )
        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(
                {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
            )

        torch.save(
            model.state_dict(),
            f"{args.out_path}/{args.model_name}/finetuned_BERT_epoch_{epoch}.model",
        )

        tqdm.write(f"\nEpoch {epoch}")

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f"Training loss: {loss_train_avg}")

        val_loss, predictions, true_vals = evaluate(
            dataloader_validation, model=model, device=device
        )
        val_f1 = f1_score_func(predictions, true_vals)
        val_p = precision_score_func(predictions, true_vals)
        val_r = recall_score_func(predictions, true_vals)
        tqdm.write(f"Validation loss: {val_loss}")
        tqdm.write(f"Precision score (macro): {val_p}")
        tqdm.write(f"Recall Score (macro): {val_r}")
        tqdm.write(f"F1 Score (macro): {val_f1}")
