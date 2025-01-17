# This does the tokenization encode and decode based on a supplied int:string vocabulary dictionary
import re


class SimpleTokenizerV1:
    # Always pass the vocab into the initializer
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}  # reversing the string:num vocab

    # returns the token ids list after encoding the new string tokens
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # Returns the text after decoding token ids list
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
