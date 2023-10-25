# Imports
import matplotlib.pyplot as plt
import torch
import torch.nn as NN
import torch.nn.functional as F
from collections import OrderedDict

# Directories path
models_path = './models/'

# Hyperparameters
block_size = 64  # The number of characters our Transformers is able to take at once
batch_size = 64
steps = 5000
lr = 3e-4
eval_iters = 200
n_embd = 32
n_heads = 4
n_blocks = 3
head_size = n_embd // n_heads
dropout = 0.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')


# For reproducibility purposes
torch.manual_seed(123445)

# Read football.txt file
with open('data/football.txt', 'r') as f:
    lines = f.readlines()
    text = ''.join(lines)

# Create our character level tokenizer
# Improvement idea: Try pre-built tokenizer like tiktoken library from OpenAI
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}

# Create helping function to encode and decode characters and indexes
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train validation split
data = torch.tensor(encode(text))
idx = int(len(data) * 0.8)
train_data = data[:idx]
val_data = data[idx:]


# Data loading function
def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - block_size, size=(batch_size,))

    x = torch.stack([data[i:i + block_size] for i in idxs])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idxs])

    return x, y


# Bigram Language Model (LM) definition
class BigramLM(NN.Module):
    def __init__(self):
        super(BigramLM, self).__init__()
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
            logits, loss = self(idx[:, -block_size:])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Head(NN.Module):
    def __init__(self, d_model, head_size):
        super(Head, self).__init__()
        self.head_size = head_size
        self.query = NN.Linear(d_model, head_size, bias=False)  # d_model, head_size
        self.key = NN.Linear(d_model, head_size, bias=False)  # d_model, head_size
        self.value = NN.Linear(d_model, head_size, bias=False)  # d_model, head_size

        self.dropout = NN.Dropout(dropout)
        # self.qkv = NN.Linear(bias=False)  # d_model, head_size*3

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


class MultiHeadAttention(NN.Module):
    """ Class implementing multi-head attention
     Simply apply n_heads in parallel using ModuleList
    """

    def __init__(self, n_heads, head_size, d_model):
        super(MultiHeadAttention, self).__init__()
        self.heads = NN.ModuleList([Head(d_model, head_size) for _ in range(n_heads)])
        self.proj = NN.Linear(n_embd, n_embd)
        self.dropout = NN.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class FFN(NN.Module):
    """ This FFN is applied to each position independently to add more computation/expressiveness to the model"""

    def __init__(self, d_model, d_inner):
        super(FFN, self).__init__()
        self.inner = NN.Linear(d_model, d_inner)
        self.out = NN.Linear(d_inner, d_model)
        self.dropout = NN.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.inner(x))
        x = self.out(x)
        x = self.dropout(x)

        return x


class Block(NN.Module):
    def __init__(self, n_head, head_size, d_model):
        super(Block, self).__init__()
        self.ln1 = NN.LayerNorm(d_model)
        self.self_heads = MultiHeadAttention(n_head, head_size, d_model)
        self.ln2 = NN.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_model)

    def forward(self, x):
        x = self.ln1(x)
        x = self.self_heads(x)
        x = self.ln2(x)
        x = self.ffn(x)

        return x


class LanguageModel(BigramLM):
    def __init__(self, n_blocks):
        super(LanguageModel, self).__init__()
        self.token_emb_table = NN.Embedding(vocab_size, n_embd)
        self.pos_end = NN.Embedding(block_size, n_embd)
        self.blocks = NN.Sequential(OrderedDict([
            (f'Block {i}', Block(n_heads, head_size, n_embd)) for i in range(n_blocks)
        ]))
        self.ln = NN.LayerNorm(n_embd)
        self.linear = NN.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb_table(idx)  # B, T, C
        pos_emb = self.pos_end(torch.arange(T, device=device))  # T, C
        x = tok_emb + pos_emb  # B, T, C
        x = self.blocks(x)  # B, T, C
        x = self.ln(x)  # B, T, C
        logits = self.linear(x)  # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


def train(model, optimizer):
    model.to(device)

    # Define training and validation loop for the BigramLM model
    eval_it = []
    train_losses = []
    val_losses = []
    print('Starting training')
    for i in range(steps):
        # Get training data
        model.train()
        train_x, train_y = get_batch('train')
        val_x, val_y = get_batch('val')

        # Send data to device used
        train_x, train_y = train_x.to(device), train_y.to(device)
        val_x, val_y = val_x.to(device), val_y.to(device)

        # Forward pass
        logits, train_loss = model(train_x, train_y)

        # Backward pass
        optimizer.zero_grad()  # Set gradient to zero for the new iteration
        train_loss.backward()  # Gradient calculation
        optimizer.step()  # Update network parameters

        # Compute validation loss
        model.eval()
        _, val_loss = model(val_x, val_y)

        if i % eval_iters == 0:
            eval_it.append(i)
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            print(f'training step: {i}, training loss: {train_loss.item():.2f}, val_loss: {val_loss.item():.2f}')

    return eval_it, train_losses, val_losses


# Create model and optimizer
model = BigramLM()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model2 = LanguageModel(n_blocks=n_blocks)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)

# eval_it_bigram, train_losses_bigram, val_losses_bigram = train(model, optimizer)
eval_it_lm, train_losses_lm, val_losses_lm = train(model2, optimizer2)

print('LM model:')
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model2.generate(idx=idx, max_new_tokens=300)[0].tolist()))
print('----------------------')

# Save model2 in case we want to use it later
torch.save(model2.state_dict(), f'{models_path}/tf_model.pt')

plt.plot(eval_it_lm, train_losses_lm, label='train loss LM')
plt.plot(eval_it_lm, val_losses_lm, label='val loss LM')
plt.legend()
plt.show()
