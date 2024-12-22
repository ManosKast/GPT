import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from tokeniser import Tokeniser

class GPTDataset(Dataset):
    '''
    A PyTorch Dataset class for the GPT model.
    '''
    def __init__(self, encodings: List[int], max_length=1024, stride=512):
        assert isinstance(encodings, list) and all(isinstance(encoding, int) for encoding in encodings), 'Encodings must be a list of integers'
        assert isinstance(max_length, int) and max_length > 0, 'Max length must be a positive integer'
        assert isinstance(stride, int) and stride > 0, 'Stride must be a positive integer'
        self.inputs = []
        self.targets = []
        
        # Final index does not include the last sequence, since it will be the target
        final_index = len(encodings) - max_length
        for i in range(0, final_index, stride):
            input_sequence = torch.tensor(encodings[i:i+max_length])
            target_sequence = torch.tensor(encodings[i+1:i+max_length+1])
            self.inputs.append(input_sequence)
            self.targets.append(target_sequence)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

def generate_dataloader(text, 
                        batch_size: int = 8, 
                        max_length: int = 1024, 
                        stride: int = 512, 
                        workers_count: int = 0):
    '''
    Receives a text, breaks it into words and generates an encoded dataloader.
    Args:
        text (str): The text to be tokenised and encoded.
        batch_size (int): The batch size for the dataloader.
        max_length (int): The maximum length of each sequence.
        stride (int): The stride for the sequences.
        workers_count (int): The number of workers for the dataloader.
    Returns:
        DataLoader: A PyTorch DataLoader object.    
    '''
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("Max length must be a positive integer")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("Stride must be a positive integer")
    if not isinstance(workers_count, int) or workers_count < 0:
        raise ValueError("Workers count must be a non-negative integer")
    tokeniser = Tokeniser()
    encodings = tokeniser.encode(text)
    dataset = GPTDataset(encodings, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers_count)
    return dataloader
