# Imports
import matplotlib.pyplot as plt
import torch

from nanoGPT.models import BigramLM, LanguageModel

# Directories path
models_path = './checkpoints/'

# Hyperparameters
block_size = 128  # The number of characters our Transformers is able to take at once
batch_size = 64
steps = 5000
lr = 4e-3
eval_iters = 200
d_model = 120
n_heads = 5
n_blocks = 5
head_size = d_model // n_heads
dropout = 0.2

if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

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


def train(model, optimizer, scheduler=None):
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

        # Reduce learning rate after some epochs
        if scheduler is not None:
            scheduler.step()

        if i % eval_iters == 0:
            eval_it.append(i)
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            print(f'training step: {i}, training loss: {train_loss.item():.2f}, val_loss: {val_loss.item():.2f}')

    return eval_it, train_losses, val_losses


# Create model and optimizer
model = BigramLM(vocab_size=vocab_size, block_size=block_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model2 = LanguageModel(
    n_blocks=n_blocks,
    block_size=block_size,
    n_heads=n_heads,
    head_size=head_size,
    d_model=d_model,
    vocab_size=vocab_size,
    device=device
)

# Print model number of parameters of language model
print(f'Number of parameters: {sum(p.numel() for p in model2.parameters())}')

# Optimizer for the language model
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)

# StepLR scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1000, gamma=0.5)

# eval_it_bigram, train_losses_bigram, val_losses_bigram = train(model, optimizer)
eval_it_lm, train_losses_lm, val_losses_lm = train(model2, optimizer2, scheduler=scheduler)

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


# Un buen ejercicio sería aumentar el tamaño del modelo, ver si se atasca y debuggear el modelo