from datasets import load_dataset
from PIL import Image
from io import BytesIO


def load_whoops():
    wmtis = load_dataset("nlphuji/wmtis-identify")["test"]
    images = wmtis["normal_image"] + wmtis["strange_image"]
    labels = [0] * len(wmtis["normal_image"]) + [1] * len(wmtis["strange_image"])
    return images, labels

def load_weird():
    weird = load_dataset("MERA-evaluation/WEIRD")["test"]

    images = []
    labels = []

    for i in range(len(weird)):
        row = weird[i]
        images.append(Image.open(BytesIO(row["inputs"]["image"]["bytes"])))
        labels.append(int(row["inputs"][f"option_{row['outputs'].lower()}"] == "странное"))

    return images, labels

def get_dataset(name):
    if name == "whoops":
        return load_whoops()
    elif name == "weird":
        return load_weird()