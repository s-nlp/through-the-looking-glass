import argparse
import os
import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from tqdm import tqdm

from utils.datasets import get_dataset


def load_model(model_name, mode):
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

    content = [{"type": "image"}]
    if mode == "prompt":
        content.append({"type": "text", "text": "Provide a brief, one-sentence descriptive fact about this image."})

    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    return model, processor, prompt


def update_states(images, hidden_states, model, processor, prompt, num_layers):
    with torch.no_grad():
        for image in tqdm(images):
            inputs = processor(prompt, image, return_tensors="pt", padding=False).to("cuda")
            output = model(**inputs, output_hidden_states=True)

            for j in range(num_layers):
                hidden_states[j].append(output.hidden_states[j][:, -1].cpu())

    result = torch.stack([torch.concat(hidden_states[i]) for i  in range(num_layers)])
    return result


def extract_hidden_states(args):
    model_name = args.model_name
    dataset = args.dataset
    mode = args.mode
    images, _ = get_dataset(dataset)
    model, processor, prompt = load_model(model_name, mode)

    num_layers = len(model.language_model.model.layers) + 1
    hidden_states = {i: [] for i in range(num_layers)}
    result = update_states(images, hidden_states, model, processor, prompt, num_layers)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{dataset}_{mode}_{model_name.split('/')[-1]}.pt")
    torch.save(result, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden states from LLaVA models.")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-vicuna-7b-hf", help="Name of the model.")
    parser.add_argument("--dataset", type=str, required=True, choices=["whoops", "weird"], help="Dataset name.")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "prompt"], help="Type of LVLM prompting.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save hidden states.")
    
    args = parser.parse_args()
    extract_hidden_states(args)
