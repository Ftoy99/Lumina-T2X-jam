import argparse
import logging
import os
import random
from copy import deepcopy
from time import time

import numpy as np
import torch
from datasets import load_dataset
from diffusers import AutoencoderKLCogVideoX
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.nextditmf import NextDiT

logger = logging.getLogger(__name__)


def ds_collate_fn(samples):
    img = []
    caption = []
    for sample in samples:
        print(samples[0])
        print(samples[1])
    return img,caption


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
    # Load the dataset
    dataset_path = "nlphuji/flickr30k"
    logger.info(f"Loading dataset {dataset_path}")
    dataset = load_dataset(dataset_path)

    dataset = prepare_dataset(dataset)

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

    dataloader = DataLoader(dataset['test'], batch_size=2, shuffle=True, collate_fn=ds_collate_fn)

    for step, x in enumerate(dataloader):
        print(f"step {step} , x {x}")
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_run", type=bool, default=False)

    args = parser.parse_known_args()[0]
    main(args)
