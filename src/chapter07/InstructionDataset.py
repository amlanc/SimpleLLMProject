from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    # The dataset class for the instruction completion task
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        #
        for entry in data:
            instruction_plus_input = self.format_input(entry)
            #
            response_text = f"\n\n### Response:\n{entry['output']}"
            #
            final_text = instruction_plus_input + response_text
            #
            self.encoded_texts.append(tokenizer.encode(final_text))

    # The __getitem__ method returns the encoded text at the given index
    def __getitem__(self, index):
        return self.encoded_texts[index]


    # The __len__ method returns the number of entries in the dataset
    def __len__(self):
        return len(self.data)
    
    # The format_input method formats the input for the model
    def format_input(self, entry):  # Now an instance method
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        input_text = (
            f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        )
        return instruction_text + input_text

