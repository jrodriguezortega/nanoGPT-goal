import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# Bigram Language Model (LM) definition
# Bigram Language Model (LM) definition
class BigramLM(nn.Module):
    def __init__(self, vocab_size, block_size):
        super(BigramLM, self).__init__()
        self.block_size = block_size
        # Create lookup table representing the bigram LM
        self.lookup_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        # x (B, T), y (B, T)
        logits = self.lookup_table(x)  # (B, T, V)
        if y is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = y.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx[:, -self.block_size:])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Head(nn.Module):
    def __init__(self, d_model, head_size, dropout=0.1):
        super(Head, self).__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size, bias=False)  # d_model, head_size
        self.key = nn.Linear(d_model, head_size, bias=False)  # d_model, head_size
        self.value = nn.Linear(d_model, head_size, bias=False)  # d_model, head_size

        self.dropout = nn.Dropout(dropout)
        # self.qkv = nn.Linear(bias=False)  # d_model, head_size*3

        # X @ QKV = B, block_size, head_size*3

    def forward(self, x, masked=True):
        # X = B, block_size, d_model
        q = self.query(x)  # B, block_size, head_size
        k = self.key(x)  # B, block_size, head_size
        v = self.value(x)  # B, block_size, head_size

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # B, block_size, block_size

        if masked:
            wei = wei.masked_fill(wei.tril() == 0, float('-inf'))

        wei = wei.softmax(-1)
        wei = self.dropout(wei)

        out = wei @ v  # B, block_size, head_size

        return out


class MultiHeadAttention(nn.Module):
    """ Class implementing multi-head attention
     Simply apply n_heads in parallel using ModuleList
    """

    def __init__(self, n_heads, head_size, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(d_model, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class FFN(nn.Module):
    """ This FFN is applied to each position independently to add more computation/expressiveness to the model"""

    def __init__(self, d_model, d_inner, dropout=0.1):
        super(FFN, self).__init__()
        self.inner = nn.Linear(d_model, d_inner)
        self.out = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.inner(x))
        x = self.out(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, n_head, head_size, d_model):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_heads = MultiHeadAttention(n_head, head_size, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_model)
        # self.out = None

    def forward(self, x):
        x = self.ln1(x)
        x = self.self_heads(x) + x
        x = self.ln2(x)
        x = self.ffn(x) + x
        # self.out = x
        return x


class LanguageModel(nn.Module):
    def __init__(self, n_blocks, d_model, n_heads, head_size, vocab_size, block_size, device='cpu'):
        super(LanguageModel, self).__init__()
        self.block_size = block_size
        self.device = device
        self.token_emb_table = nn.Embedding(vocab_size, d_model)
        self.pos_end = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(OrderedDict([
            (f'Block {i}', Block(n_heads, head_size, d_model)) for i in range(n_blocks)
        ]))
        self.ln = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

        # self.out = None  # For debugging purposes, delete it afterwards

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb_table(idx)  # B, T, C
        pos_emb = self.pos_end(torch.arange(T, device=self.device))  # T, C
        x = tok_emb + pos_emb  # B, T, C
        x = self.blocks(x)  # B, T, C
        x = self.ln(x)  # B, T, C
        # self.out = x
        logits = self.linear(x)  # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx[:, -self.block_size:])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

