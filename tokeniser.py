from typing import List
import re
import tiktoken

#TODO: Write my own byte-pair encoder
class Tokeniser:
    def __init__(self):
        self.tokeniser = tiktoken.get_encoding('gpt2')

    def encode(self, text: str) -> List[int]:
        assert isinstance(text, str), 'Text must be a string'
        encoded_text = self.tokeniser.encode(text, allowed_special={'<|endoftext|>'})
        return encoded_text
    
    def decode(self, tokens: List[int]) -> str:
        assert isinstance(tokens, list) and all(isinstance(token, int) for token in tokens), 'Tokens must be integers'
        decoded_text = self.tokeniser.decode(tokens)
        return decoded_text
        

if __name__ == '__main__':
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()

    tokeniser = Tokeniser()
    encoded_text = tokeniser.encode(text)
    print(encoded_text[:50])
    decoded_text = tokeniser.decode(encoded_text)
    print(decoded_text[:50])