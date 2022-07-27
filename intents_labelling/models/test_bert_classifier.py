import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

from intents_labelling.data_loaders import load_labelled_orcas
from intents_labelling.models.evaluation import (
    evaluate,
    f1_score_func,
    accuracy_per_class,
    precision_score_func,
    recall_score_func,
    read_labels,
)
from intents_labelling.models.preprocessing import (
    remove_punctuation,
    query_plus_url,
    get_domains,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pred(test_data):
    prediction_list = []
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
        prediction_list.append((sm, loss))
    return prediction_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="xtremedistil", type=str)
    parser.add_argument("--model_params", default="query_url_2M-64-level2", type=str)
    parser.add_argument("--infile", default="data/test/orcas_test.tsv", type=str)
    parser.add_argument("--model_path", default="models/xbert/", type=str)
    parser.add_argument(
        "--input_features", default="query", choices=("query", "query_url"), type=str
    )

    args = parser.parse_args()

    labels_file = "labels.json"

    df = load_labelled_orcas(data_path=args.infile)
    label_column = "label_manual"

    if args.input_features == "query":
        data_column = "query"
    elif args.input_features == "query_url":
        data_column = "query_url"
    else:
        raise ValueError("data_column can be only query or query_url")

    if args.model_name == "xtremedistil":
        model_name = "microsoft/xtremedistil-l6-h384-uncased"
    elif args.model_name == "bert":
        model_name = "bert-base-uncased"

    g_d = get_domains(df, "url")
    d_p = remove_punctuation(g_d, "domain_names")
    df = query_plus_url(d_p, "query", "domain_names")

    label_dict = read_labels(
        infile=f"{args.model_path}/{args.model_name}_{args.model_params}/{labels_file}"
    )

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    df["label"] = df[label_column].replace(label_dict)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
    )

    model.to(device)

    # %%

    for model_version in range(1, 10):
        print(f"{model_version=}")
        model.load_state_dict(
            torch.load(
                f"{args.model_path}/{args.model_name}_{args.model_params}/finetuned_BERT_epoch_{model_version}.model",
                map_location=torch.device("cpu"),
            )
        )

        batch_size = 32

        encoded_data_val = tokenizer.batch_encode_plus(
            df[data_column].values,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
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

        _, predictions, true_vals = evaluate(
            dataloader_validation, model=model, device=device
        )

        print(accuracy_per_class(predictions, true_vals, label_dict=label_dict))
        print(f"f1 = {f1_score_func(predictions, true_vals)}")
        print(f"prec = {precision_score_func(predictions, true_vals)}")
        print(f"rec = {recall_score_func(predictions, true_vals)}")

    pr = pred([("medical assistance", "Factual"), ("what is my name", "Factual")])
    print(pr)
    print(label_dict)
