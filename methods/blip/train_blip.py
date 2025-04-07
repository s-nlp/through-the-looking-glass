import argparse
import os
import torch

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import BlipProcessor, Blip2ForConditionalGeneration, AdamW

from utils.finetuning import ImageCaptioningDataset, set_seed, CONFIG, fetch_scheduler, run_training, inference
from utils.datasets import get_dataset


def train_model(args):
    set_seed(CONFIG["seed"])

    CONFIG["model_name"] = args.model_name
    CONFIG["processor"] = BlipProcessor.from_pretrained(CONFIG["model_name"])
    train_images, train_labes = get_dataset(args.tr_dataset)
    test_images, test_labels = get_dataset(args.ts_dataset)

    model = Blip2ForConditionalGeneration.from_pretrained(CONFIG["model_name"], torch_dtype=torch.bfloat16)

    train_dataset = ImageCaptioningDataset(train_images, train_labes, CONFIG["processor"])
    valid_dataset = ImageCaptioningDataset(test_images, test_labels, CONFIG["processor"])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG["train_batch_size"])
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=CONFIG["valid_batch_size"])

    model.to(CONFIG["device"])

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = fetch_scheduler(optimizer)

    model, _, _ = run_training(model, optimizer, scheduler, 
                               train_loader, valid_loader,
                               num_epochs=CONFIG["epochs"])

    results = inference(model, test_images, device=CONFIG["device"])
    accuracy_score(test_labels, results)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{args.model_name.split("/")[-1]}_{args.tr_dataset}->{args.ts_dataset}.pt")
    model.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BLIP model.")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-flan-t5-xl", help="Name of the model.")
    parser.add_argument("--tr_dataset", type=str, required=True, choices=["whoops", "weird"], help="Train dataset name.")
    parser.add_argument("--ts_dataset", type=str, required=True, choices=["whoops", "weird"], help="Test dataset name.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save trained model.")
    
    args = parser.parse_args()
    train_model(args)
