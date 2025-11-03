import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from GPTDataset import generate_dataloader
from typing import List

import gc

from model import GPTModel
from tokeniser import Tokeniser

def text_to_token(text, tokeniser: Tokeniser):
    tokenised = tokeniser.encode(text)
    return torch.tensor(tokenised).unsqueeze(0)

def token_to_text(tokens, tokeniser):
    flattened = tokens.squeeze(0)
    return tokeniser.decode(flattened.tolist())

def generate_print_sample(model, context: str, tokeniser: Tokeniser):
    model.eval()
    encoded_tokens = text_to_token(context, tokeniser)
    # TODO: Maybe remove
    with torch.no_grad():
        tokens = generate_text(model, encoded_tokens, 30)
    print(token_to_text(tokens, tokeniser))
    model.train()


def generate_text(model: GPTModel, context: torch.Tensor, words_count: int, temperature: int = 1.4, topk: int = 5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context = context.to(device)
    for _ in range(words_count):
        with torch.no_grad():
            logits = model(context)
        predicted_logits = logits[:, -1, :]
        vocab_size = predicted_logits.size(-1)
        if not isinstance(topk, int):
            raise TypeError('topk must be an integer')
        if topk <= 0 or topk > vocab_size:
            raise ValueError(f'topk must be in the range [1, {vocab_size}]')
        if temperature <= 0:
            raise ValueError('temperature must be greater than zero')
        top_k_logits, _ = torch.topk(input=predicted_logits, k=topk)
        min_value = top_k_logits[:, -1]
        masked_logits = torch.where((predicted_logits < min_value), torch.tensor(-torch.inf).to(logits.device), predicted_logits)

        masked_logits = masked_logits / temperature
        probability_distribution = torch.softmax(masked_logits, dim=-1)
        next_token = torch.multinomial(probability_distribution, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
    return context

def compute_loss_batch(model: GPTModel, inputs: torch.Tensor, targets: torch.tensor, device) -> torch.Tensor:
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    loss = nn.functional.cross_entropy(input=logits.flatten(0, 1), target=targets.flatten())
    return loss   

def compute_loss_loader(model, data_loader, device, num_batches=None) -> float:
    total_loss = 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = compute_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break            
    return total_loss / num_batches 

def evaluate_model(model: GPTModel, data_loader: DataLoader, validation_loader: DataLoader, device: torch.device):
    model.eval()
    with torch.no_grad():
        train_loss = compute_loss_loader(model, data_loader, device)
        valid_loss = compute_loss_loader(model, validation_loader, device)
    return train_loss, valid_loss

def train_model(model, 
                epochs: int,
                optimiser: torch.optim.AdamW,
                train_loader: DataLoader,
                validation_loader: DataLoader,
                device: torch.device,
                evaluation_frequency: int) -> int:
    print('Training model...')
    tokeniser = Tokeniser()
    step = -1
    for epoch in range(epochs):
        model.train()
        for input, target in train_loader:
            model.zero_grad()
            loss = compute_loss_batch(model, input, target, device)
            loss.backward()
            optimiser.step()
            step += 1
            if step % evaluation_frequency == 0:
                train_loss, validation_loss = evaluate_model(model, train_loader, validation_loader, device)
                track_tokens_seen = (step * input.size(0))
                print(f'Ep {epoch+1}:\nTrain loss: {train_loss}, Validation loss: {validation_loss}')
                start_context = 'How'
                generate_print_sample(model, start_context, tokeniser)

                # Append training progress to a file
                with open('training_progress.txt', 'a') as f:
                    f.write('Step,Train Loss,Validation Loss,Tokens Seen\n')
                    f.write(f'{train_loss},{validation_loss},{track_tokens_seen}\n')

        # Save the model parameters
        torch.save(model.state_dict(), f'./models/gpt_model_epoch_{epoch+1}.pth')


def checkpoint(model: GPTModel, path: str):
    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    tokeniser = Tokeniser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voc_size, embedding_dimension, layers_count, attention_heads, context_length = 50_257, 1_280, 36, 20, 1_024
    model = GPTModel(voc_size, embedding_dimension, context_length, attention_heads, layers_count, 0.0, False)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    file = 'test.txt'
    with open(file, 'r') as f:
        text = f.read()
    train_loader = generate_dataloader(text, batch_size=4, max_length=256, stride=128, workers_count=0, device=device)
    validation_loader = generate_dataloader(text, batch_size=4, max_length=256, stride=128, workers_count=0, device=device)
    checkpoint_path = './models/gpt_model_epoch_6.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    train_model(model, 20, optimiser, train_loader, validation_loader, device, 10)
