import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_hf_checkpoint(checkpoint: str):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model, tokenizer


def extract_avg_pooled_features(data, nli_model, tokenizer):
    features = []

    print("Extracting and pooling the features...")
    
    for n in tqdm(data):
        f = tokenizer(list(n), padding=True, return_tensors="pt").to(nli_model.device)
        with torch.no_grad():
            outputs = nli_model.deberta(**f, output_hidden_states=True)
            # outputs = nli_model.model(**f, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
            attention_mask = f["attention_mask"]
            masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            sum_hidden_states = masked_hidden_states.sum(dim=1)
            mask_sum = attention_mask.sum(dim=1, keepdim=True)
            pooled_hidden_states = sum_hidden_states / mask_sum
        features.append(pooled_hidden_states)

    features = torch.stack(features)

    print(f"Extracted a tensor of features with the following shape: {features.shape}")

    return features


def train_on_fold(
    train_dataloader, test_dataloader, epoch_cnt, lr, nli_model, model_class
):
    model = model_class(nli_model_dim=nli_model.config.hidden_size).cuda()

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []

    test_acc = 0

    for epoch in range(epoch_cnt):
        train_preds = []
        train_trues = []
        epoch_train_losses = []

        model.train()
        for x, y in train_dataloader:
            pred = model(x.cuda())
            loss = loss_function(pred, y.view(-1, 1).cuda() * 1.0)
            epoch_train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_preds.extend([p[0].item() for p in pred])
            train_trues.extend(y)
        train_losses.append(np.mean(epoch_train_losses))

    test_trues = []
    test_preds = []

    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            pred = F.sigmoid(model(x.cuda()))
            test_preds.extend([p[0].item() for p in pred])

            test_trues.extend(y)

    test_acc = accuracy_score([p > 0.5 for p in test_preds], test_trues)
    return test_acc, model


def load_whoops_data(extracted_facts_path):
    whoops_data = read_jsonl(Path(extracted_facts_path))

    whoops_normal = [w["normal"] for w in whoops_data]
    whoops_strange = [w["strange"] for w in whoops_data]

    data = whoops_normal + whoops_strange
    labels = np.array([1] * len(whoops_normal) + [0] * len(whoops_strange))

    return data, labels


def load_weird_data(extracted_facts_path):
    weird_dataset = load_dataset("MERA-evaluation/WEIRD")["test"]
    data = read_jsonl(Path(extracted_facts_path))
    labels = [get_weird_label(row) for row in weird_dataset]
    return data, labels
