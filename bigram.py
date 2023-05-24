# Imports
import matplotlib.pyplot as plt
import torch
import torch.nn as NN
import torch.nn.functional as F

# Hyperparameters
block_size = 8  # The number of characters our Transformers is able to take at once
batch_size = 16
steps = 10000
lr = 1e-3
eval_iters = 200

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
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Create model and optimizer
model = BigramLM()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

plt.plot(eval_it, train_losses, label='train loss')
plt.plot(eval_it, val_losses, label='val loss')
plt.legend()
plt.show()

# Generation of sequence of characters AFTER model training
idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(idx=idx, max_new_tokens=300)[0].tolist()))
