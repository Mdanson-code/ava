import re
from typing import List, Dict, Optional

class Tokenizer:
    """
    A simple word-level tokenizer for the transformer model.
    """
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        # Initialize with special tokens
        for token in self.special_tokens:
            self.add_word(token)
    
    def add_word(self, word: str) -> int:
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.vocab_size += 1
            return idx
        return self.word2idx[word]
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 50000) -> None:
        """Build vocabulary from a list of texts."""
        word_freq = {}
        
        # Count word frequencies
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add most frequent words to vocab
        for word, _ in sorted_words[:max_vocab_size - len(self.special_tokens)]:
            self.add_word(word)
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization of text into words."""
        # Convert to lowercase and split on whitespace and punctuation
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        return words
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Convert text to sequence of token IDs."""
        words = self._tokenize(text)
        
        # Add BOS and EOS tokens
        words = [self.bos_token] + words + [self.eos_token]
        
        # Convert words to indices
        ids = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        
        # Pad or truncate to max_length if specified
        if max_length is not None:
            if len(ids) < max_length:
                # Pad with PAD token
                ids = ids + [self.word2idx[self.pad_token]] * (max_length - len(ids))
            else:
                # Truncate
                ids = ids[:max_length-1] + [self.word2idx[self.eos_token]]
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Convert sequence of token IDs back to text."""
        words = []
        for idx in ids:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word == self.eos_token:
                    break
                if word not in self.special_tokens:
                    words.append(word)
        
        return " ".join(words)
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to a file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {int(k): v for k, v in self.idx2word.items()},
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Tokenizer':
        """Load tokenizer from a file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.pad_token = data['pad_token']
        tokenizer.unk_token = data['unk_token']
        tokenizer.bos_token = data['bos_token']
        tokenizer.eos_token = data['eos_token']
        
        return tokenizer
