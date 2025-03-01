from datasets import load_dataset

# Load the Flickr30k dataset
dataset = load_dataset("nlphuji/flickr30k")

# View a sample from the dataset
print(dataset)  # Access the first sample in the training set
