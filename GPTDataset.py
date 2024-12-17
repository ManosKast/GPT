import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from tokeniser import Tokeniser

class GPTDataset(Dataset):
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
    

def generate_dataloader(text, batch_size=8, max_length=1024, stride=512, workers_count=0):
    tokeniser = Tokeniser()
    encodings = tokeniser.encode(text)
    dataset = GPTDataset(encodings, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=workers_count)
    return dataloader
    
if __name__ == '__main__':
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()

    dataloader = generate_dataloader(text, batch_size=1, max_length=4, stride=1)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)