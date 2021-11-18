import torch
from transformers import BertForSequenceClassification, BertTokenizer
from intents_labelling.data_loaders import load_labelled_orcas


def predict():
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False,
)

model.to(device)

# %%

model.load_state_dict(torch.load("finetuned_BERT_epoch_5.model"))

# %%

_, predictions, true_vals = evaluate(dataloader_validation)

# %%

accuracy_per_class(predictions, true_vals)
f1_score_func(predictions, true_vals)

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
