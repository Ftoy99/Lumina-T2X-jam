from datasets import load_dataset

# Load the Flickr30k dataset
dataset = load_dataset("flickr30k")

# Iterate over a few samples
for idx, sample in enumerate(test_data):
    # Print the image and its corresponding caption
    print(f"Image: {sample['image']}")
    print(f"Caption: {sample['caption']}")
    print(f"Sentence ID: {sample['sentids']}")
    print(f"Image ID: {sample['img_id']}")
    print(f"Filename: {sample['filename']}")

    # Break after the first 3 samples (just for demonstration)
    if idx == 2:
        break