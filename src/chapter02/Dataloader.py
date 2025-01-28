import string

import tiktoken
from torch.utils.data import DataLoader
from src.chapter02.GPTDatasetV1 import GPTDatasetV1


class Dataloader:
    def __init__(self, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=False, num_workers=0):
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.txt = ""

    # Now we need to create a data loader that iterates over the input dataset
    # and returns the input and targets as PyTorch tensors
    def create_dataloader_v1(self, txt: string):
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset using GPTDatasetV1
        dataset = GPTDatasetV1(txt, tokenizer, self.max_length, self.stride)

        # Create dataloader
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                drop_last=self.drop_last,
                                num_workers=self.num_workers)

        return dataloader
