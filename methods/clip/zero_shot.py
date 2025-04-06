from io import BytesIO

import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def classify_image(image):
    inputs = processor(
        text=["Unusual image", "Normal image"],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(
        dim=1
    )  # we can take the softmax to get the label probabilities
    return probs.argmax(-1).item()


def classify_image_siglip(image):
    inputs = processor(
        text=["Unusual image", "Normal image"],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image)
    return probs.argmax(-1).item()


checkpoints = [
    "openai/clip-vit-base-patch32",
    "google/siglip-so400m-patch14-384",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
]

print("Running on WHOOPS!")
dataset = load_dataset("nlphuji/wmtis-identify")["test"]

for checkpoint in checkpoints:
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to("cuda")
    processor = AutoProcessor.from_pretrained(checkpoint)

    normal_answers = []
    strange_answers = []

    for sample in tqdm(dataset):
        if "siglip" in checkpoint:
            strange_answers.append(
                classify_image_siglip(sample["strange_image"].convert("RGB"))
            )
            normal_answers.append(
                classify_image_siglip(sample["normal_image"].convert("RGB"))
            )
        else:
            strange_answers.append(
                classify_image(sample["strange_image"].convert("RGB"))
            )
            normal_answers.append(classify_image(sample["normal_image"].convert("RGB")))

    score = accuracy_score(
        strange_answers + normal_answers,
        [0] * len(strange_answers) + [1] * len(normal_answers),
    )
    print(f"WHOOPS! Zero shot {checkpoint}: {round(score * 100, 2)}")

print("Running on WEIRD")


def get_weird_label(row):
    correct_option = "option_a" if row["outputs"] == "A" else "option_b"
    return 1 if row["inputs"][correct_option] == "нормальное" else 0


weird_dataset = load_dataset("MERA-evaluation/WEIRD")["test"]
images = [Image.open(BytesIO(t["inputs"]["image"]["bytes"])) for t in weird_dataset]
labels = [get_weird_label(row) for row in weird_dataset]

for checkpoint in checkpoints:
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to("cuda")
    processor = AutoProcessor.from_pretrained(checkpoint)

    answers = []

    for image in tqdm(images):
        if "siglip" in checkpoint:
            answers.append(classify_image_siglip(image.convert("RGB")))
        else:
            answers.append(classify_image(image.convert("RGB")))

    score = accuracy_score(answers, labels)
    print(f"WEIRD Zero shot {checkpoint}: {round(score * 100, 2)}")
