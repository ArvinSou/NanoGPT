# disable cpu fallback:
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters:
batch_size = 32
block_size = 256
# for a T4 gpu, the model starts overfitting after ~3000 iterations
max_iters = 3000
eval_interval = 200 # higher for faster training time
learning_rate = 3e-4
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
# for mac:
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
print("Device being used:", 'mps' if torch.backends.mps.is_available() else 'cpu')

eval_iters = 20 # instead of 200 for faster training time
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for s, i in stoi.items()}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

# train & test splits:
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading:
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ One Head Of Self-Attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size ---> (batch, time-step, channels)
        # output of size ---> (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # compute attention scores ("affinities"):
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # replace zeros with -infs:
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, -1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values:
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class MultiHeadAttention(nn.Module):
    """ Multiple Heads Of Self-Attention In Parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        # final layer norm:
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of ints
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens:
            idx_cond = idx[:, -block_size:]
            # get the predictions:
            logits, loss = self(idx_cond)
            # focus only on the last tiem step:
            logits = logits[:, -1, :] # becomes (B, T)
            # apply softmax to get probs:
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample:
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

def f():
    # create a PyTorch optimizer:
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data:
        xb, yb = get_batch('train')

        # evaluate the loss:
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # save the model parameters:
    model_save_path = 'shakespeare_char_model.pth'
    torch.save(m.state_dict(), model_save_path)

#f()
# generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
MODEL_PATH = "shakespeare_char_model_T4.pth"
TEMPERATURE = 0.85
TOP_K = 40
TOP_P = 0.92
MAX_NEW_TOKENS = 10000

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print("Model Loaded Successfully On ", device)

@torch.no_grad()
def generate(idx: str, max_new_tokens=200, temperature=0.8, top_k=None, top_p=None):
    idx = encode(idx)
    idx = torch.tensor(idx, dtype=torch.long, device=device)
    idx = idx.unsqueeze(0) # add the batch dimension

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        # forward pass:
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        
        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = -float('Inf')

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove]


        # sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    #return decode(idx[0].tolist())
    return idx[0].tolist()

prompt = " "
print("\nPrompt: ", prompt)

generated = generate(
    prompt,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P
)

generated = decode(generated)

print("\nGenerated:\n" + "="*50)
print(generated)
print("="*50)
