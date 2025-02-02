import tiktoken
import torch
from torch.utils.data import Dataset
import pandas as pd



# This class identifies the longest sequence in the training dataset, encodes
# the text messages, and ensures that all other sequences are padded with a
# padding token to match the length of the longest sequence.
#
# NOTE: This class is to be used against train, validate and test csv files not
# against the main tsv file.
class SpamDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_length, pad_token_id=50256):
        if (csv_file.endswith('.tsv')):
            self.data = pd.read_csv(csv_file, sep = "\t", header = None, names = ["Label", "Text"])
        else:
            self.data = pd.read_csv(csv_file)
        print(f"Shape of {csv_file}: {self.data.shape}")
        
        # Pre tokenize
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
            
        else:
            self.max_length = max_length
            # Truncate each line if longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        
        # Then Pad as needed to match max_length
        self.encoded_texts = [
            encoded_text +
            [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]
    
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_text_length = len(encoded_text)
            if encoded_text_length > max_length:
                max_length = encoded_text_length
        return max_length
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    
    
def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SpamDataset(csv_file="../../data/sms_spam_collection/SMSSpamCollection.tsv",
                          tokenizer=tokenizer,
                          max_length=None,
                          pad_token_id=50256)
    print("Dataset Max Length: ", dataset.max_length)

if __name__ == '__main__':
    main()
        
        