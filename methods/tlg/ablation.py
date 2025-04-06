import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import BATCH_SIZE, LEARNING_RATE, SEED, DatasetType, LEARNING_RATE
from dataset import TextBinaryDataset
from model import AttentionPoolingModel
from utils import (
    extract_avg_pooled_features,
    load_hf_checkpoint,
    load_weird_data,
    load_whoops_data,
    train_on_fold,
)


def run_ablation(
    encoder_checkpoint_path: str, extracted_facts_path: str, dataset_type: DatasetType
) -> None:
    if dataset_type == DatasetType.WEIRD:
        data, labels = load_weird_data(extracted_facts_path)
    if dataset_type == DatasetType.WHOOPS:
        data, labels = load_whoops_data(extracted_facts_path)

    nli_model, tokenizer = load_hf_checkpoint(encoder_checkpoint_path)
    features = extract_avg_pooled_features(data, nli_model, tokenizer)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    BEST_FOLD_ACC = 0

    for epoch_cnt in tqdm(range(5, 100, 5)):
        fold_accs = []

        for i, (train_index, test_index) in enumerate(kf.split(features)):
            train_x = features[train_index]
            train_y = labels[train_index]

            test_x = features[test_index]
            test_y = labels[test_index]

            train_dataset = TextBinaryDataset(texts=train_x, labels=train_y)
            test_dataset = TextBinaryDataset(texts=test_x, labels=test_y)

            train_dataloader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False
            )

            last_test_acc, model = train_on_fold(
                train_dataloader,
                test_dataloader,
                epoch_cnt=epoch_cnt,
                lr=LEARNING_RATE,
                nli_model=nli_model,
                model_class=AttentionPoolingModel,
            )
            fold_accs.append(last_test_acc)

        if np.mean(fold_accs) > BEST_FOLD_ACC:
            BEST_FOLD_ACC = np.mean(fold_accs)
            print(
                f"New best setting: {LEARNING_RATE=} {BATCH_SIZE=} {epoch_cnt=} fold_accs={[round(f, 3) for f in fold_accs]} mean_acc={round(np.mean(fold_accs) * 100, 2)}"
            )
