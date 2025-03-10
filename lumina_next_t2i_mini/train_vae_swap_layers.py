import argparse
import gc
import json
import logging
import os
import random

import numpy
import numpy as np
import torch
from PIL import Image
import cv2
from diffusers import AutoencoderKLCogVideoX
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from transport_mf import training_losses
from models.nextditmf import NextDiT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_memory = torch.cuda.get_device_properties(device).total_memory
vae_scale = 0.13025
torch.set_float32_matmul_precision('high')


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

        text_input_ids = text_input_ids.to("cpu")
        prompt_masks = prompt_masks.to("cpu")

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]
    prompt_embeds = prompt_embeds.to(device)
    prompt_masks = prompt_masks.to(device)
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
        .cpu()
    )
    cap_feat_dim = text_encoder.config.hidden_size

    # Load vae
    logger.info(f"Creating vae {dataset_path}")
    vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16).to(
        "cuda")
    torch.cuda.empty_cache()
    # Creating model
    logger.info(f"Creating model {dataset_path}")
    model = NextDiT(patch_size=2, dim=2304, n_layers=24, n_heads=32, n_kv_heads=8, cap_feat_dim=cap_feat_dim,
                    qk_norm=True)
    model.to(device)
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
            f"custom_ckpt/1.safetensors",
        )
    model.load_state_dict(ckpt, strict=True)

    # TODO Remove some layers for memory

    # Optimizer
    logger.info(f"Creating optimizer")
    # Train only vae conv layers
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze
    for param in model.vae_out.parameters():
        param.requires_grad = True
    for param in model.vae_in.parameters():
        param.requires_grad = True

    parameters_to_train = [
        {'params': model.vae_out.parameters(), 'lr': 1e-4},
        {'params': model.vae_in.parameters(), 'lr': 1e-4},
    ]

    opt = torch.optim.AdamW(parameters_to_train, lr=1e-4)

    logger.info("Setting model to training")
    model.train()

    max_steps = 50000
    accumulation_steps = 32
    logger.info(f"Training for {max_steps} steps of 32")

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=ds_collate_fn, pin_memory=False)

    # Create scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for step, data in enumerate(dataloader):
        if step >= max_steps:
            break  # Exit the inner loop if we've reached the steps per epoch
        # Logging
        logger.info(f"Step [{step}]")

        logger.info("Empty Cache")
        torch.cuda.empty_cache()
        gc.collect()

        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - reserved_memory
        logger.info(f"Total memory: {total_memory / 1e9:.2f} GB")
        logger.info(f"Allocated memory: {allocated_memory / 1e9:.2f} GB")
        logger.info(f"Reserved memory: {reserved_memory / 1e9:.2f} GB")
        logger.info(f"Available memory (estimated): {available_memory / 1e9:.2f} GB")

        images, caps = data

        frames_resized = np.array(
            [cv2.resize(numpy.array(frame), (512, 512)) for frame in images])  # Resize all frames
        #  batch_size, num_channels, num_frames, height, width = x.shape
        frames_tensor = torch.from_numpy(frames_resized).permute(3, 0, 1, 2).unsqueeze(0)
        frames_tensor = frames_tensor.to(dtype=torch.float16, non_blocking=True) / 127.5 - 1  # Normalize

        with torch.no_grad():
            frames_tensor = frames_tensor.to(dtype=torch.float16, device="cuda")
            latent = vae.encode(frames_tensor).latent_dist.sample().mul_(vae_scale)
            assert not torch.isnan(latent).any(), "NaN detected in latent!"
            latent = latent.to(device)

        del frames_tensor
        torch.cuda.empty_cache()
        gc.collect()

        with torch.no_grad():
            cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, 0.3)  # Empty prompts 0.3 of the time

        loss_item = 0.0

        model_kwargs = dict(cap_feats=cap_feats.contiguous(), cap_mask=cap_mask.contiguous())

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.float32):
            loss_dict = training_losses(model, latent, latent, model_kwargs)

        loss = loss_dict["loss"].sum()
        scaler.scale(loss).backward()  # Scale loss and backpropagate
        logger.info(f"Loss is {loss} for step {step}")

        if (step + 1) % accumulation_steps == 0:
            logger.info("Stepping optimizer")
            scaler.unscale_(opt)  # Unscale gradients to FP32
            scaler.step(opt)  # Step the optimizer with FP32 gradients
            scaler.update()  # Update the scaler after the optimizer step
            opt.zero_grad()  # Zero gradients for next iteration

        loss_item += loss.item()
        # Save the model every 200 steps (or adjust as needed)
        if (step + 1) % 200 == 0:
            save_file(model.state_dict(), f'custom_ckpt/1.safetensors')
            logger.info(f"State dict keys: {list(model.state_dict().keys())}")  # Print all keys
            logger.info(f"Saved model checkpoint at steps {step + 1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_run", type=bool, default=False)
    args = parser.parse_known_args()[0]
    main(args)
