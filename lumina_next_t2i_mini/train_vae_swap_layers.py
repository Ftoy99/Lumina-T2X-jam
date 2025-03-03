import argparse
import json
import logging
import os
import random
from copy import deepcopy
from time import time

import numpy
import numpy as np
import torch
from PIL import Image
import cv2
from diffusers import AutoencoderKLCogVideoX
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from transport_mf import training_losses
from models.nextditmf import NextDiT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageTextDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.file_pairs = []

        # Find all jpg images and their corresponding json files
        for file in os.listdir(folder):
            if file.endswith(".jpg"):
                img_path = os.path.join(folder, file)
                json_path = os.path.join(folder, file.replace(".jpg", ".json"))
                if os.path.exists(json_path):
                    self.file_pairs.append((img_path, json_path))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path, json_path = self.file_pairs[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load JSON and get the prompt
        with open(json_path, "r") as f:
            metadata = json.load(f)
            prompt = metadata.get("prompt", "")

        return {"image": image, "prompt": prompt}


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    return logging.getLogger(__name__)


def ds_collate_fn(batch):
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    return images, prompts


def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        text_input_ids = text_input_ids.to(device)
        prompt_masks = prompt_masks.to(device)

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


def prepare_dataset(dataset):
    valid_items = []
    min_width = 512
    min_height = 512
    for item in dataset["test"]:
        img = item['image']  # The image is already a PIL Image
        if img.width >= min_width and img.height >= min_height:
            valid_items.append(item)
    return valid_items


def main(args):
    torch.cuda.set_device(0)

    # Create logger
    logger = create_logger("logs")

    # Load the dataset
    dataset_path = "/mnt/jimmys/dataset_jacky/data"
    logger.info(f"Loading dataset {dataset_path}")
    dataset = ImageTextDataset(dataset_path)
    logger.info(f"Size of dataset after preprocessing is {len(dataset)}")

    # logger.info(f"Preprocessing dataset")
    # dataset = prepare_dataset(dataset)

    # Load the tokenizers
    tokenizer_path = "google/gemma-2b"
    logger.info(f"Loading tokenizer {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "right"

    # Load the encoder
    logger.info(f"Loading text encoder {tokenizer_path}")
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            tokenizer_path,
            torch_dtype=torch.bfloat16,
        )
        .get_decoder()
        .cuda()
    )
    cap_feat_dim = text_encoder.config.hidden_size

    # Load vae
    logger.info(f"Creating vae {dataset_path}")
    vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16).to(
        "cuda")

    # Creating model
    logger.info(f"Creating model {dataset_path}")
    model = NextDiT(patch_size=2, dim=2304, n_layers=24, n_heads=32, n_kv_heads=8, cap_feat_dim=cap_feat_dim)
    model.to(device)
    model.half()
    torch.cuda.synchronize()

    logger.info(f"Loading model model {dataset_path}")
    if args.first_run:
        ckpt = load_file(
            f"Lumina-Next-SFT/consolidated_ema.00-of-01.safetensors",
        )
        # Extend the first layer with the normal weights
        logger.info(f"Extending x_embedder dimensions")
        with torch.no_grad():
            model.x_cat_emb.weight[:, :model.x_embedder.in_features] = model.x_embedder.weight
            model.x_cat_emb.bias.copy_(model.x_embedder.bias)

            motion_dim_start = model.x_embedder.in_features
            model.x_cat_emb.weight[:, motion_dim_start:].zero_()
    else:
        ckpt = load_file(
            f"custom_ckpt/consolidated_ema.00-of-01.safetensors",
        )
    model.load_state_dict(ckpt, strict=False)
    logger.info(f"Creating ema model")
    model_ema = deepcopy(model)

    # Optimizer
    logger.info(f"Creating optimizer")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Encode captions
    # with torch.no_grad():
    #     cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)

    logger.info("Setting model to training")
    model.train()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    max_steps = 100
    logger.info(f"Training for {max_steps:,} steps...")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=ds_collate_fn)

    for step, data in enumerate(dataloader):
        logger.info(f"Step [{step}]")
        images, caps = data

        with torch.no_grad():
            frames_resized = np.array(
                [cv2.resize(numpy.array(frame), (512, 512)) for frame in images])  # Resize all frames
            print(f"np array frames shape {frames_resized.shape}")  # np array frames shape (708, 512, 512, 3)
            #  batch_size, num_channels, num_frames, height, width = x.shape
            frames_tensor = torch.tensor(frames_resized).permute(3, 0, 1, 2).unsqueeze(
                0)
            frames_tensor = frames_tensor.to(torch.float16).to("cuda") / 127.5 - 1  # Normalize
            print(f"frames_tensor shape {frames_tensor.shape}")
            latent = vae.encode(frames_tensor).latent_dist.sample()
            logger.info(f"Frames shapes {latent.shape}")

        with torch.no_grad():
            cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, 0.3)  # Empty prompts 0.3 of the time
            print(f"cap_feats shape {cap_feats.shape}")
            print(f"cap_mask shape {cap_mask.shape}")

        loss_item = 0.0
        opt.zero_grad()
        model_kwargs = dict(cap_feats=cap_feats, cap_mask=cap_mask)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            latent = latent.repeat(1,1,5,1,1)
            loss_dict = training_losses(model, latent, latent, model_kwargs)
            loss = loss_dict["loss"].sum()
            loss_item += loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_run", type=bool, default=False)

    args = parser.parse_known_args()[0]
    main(args)
