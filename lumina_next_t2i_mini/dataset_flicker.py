import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import functools

patch_size = 8 * model_patch_size
max_num_patches = round((args.image_size / patch_size) ** 2)
logger.info(f"Limiting number of patches to {max_num_patches}.")
crop_size_list = generate_crop_size_list(max_num_patches, patch_size)


def var_center_crop(pil_image, crop_size_list, random_top_k=4):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


# Define your image transformations
image_transform = transforms.Compose(
    [
        transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)

# Load the dataset
dataset = load_dataset("nlphuji/flickr30k")


# Apply transformation to the 'image' column
def apply_transforms(example):
    image = Image.open(example['image']).convert('RGB')
    example['image'] = image_transform(image)  # Apply the transformation
    return example


# Apply transformation to the entire dataset
dataset = dataset.map(apply_transforms)


# You can use the DataLoader as usual for batching
def collate_fn(batch):
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    return torch.stack(images), captions


# Create DataLoader for batching
batch_size = 32
dataloader = DataLoader(dataset['test'], batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# Example: Iterate through the DataLoader
for batch_idx, (images, captions) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"Images: {images.shape}")
    print(f"Captions: {captions[:2]}")  # Print the first two captions in the batch
    break  # Just for demonstration, to print the first batch
