import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================================
# 《Let's build GPT》完整版——gpt-dev.ipynb 跟到「单头自注意力」后，
# 这份文件把剩下的零件全部组装完成：
#   数据：tiny shakespeare，字符级词表 65
#   模型：6 层 Transformer Block、384 维、6 头，约 10.8M 参数
#   训练：AdamW 5000 步；末尾从换行符开始采样生成（训练产物见 more.txt）
# =====================================================================

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?（并行处理多少条序列）
block_size = 256 # what is the maximum context length for predictions?（上下文窗口：最多回看 256 个字符）
max_iters = 5000              # 训练步数
eval_interval = 500           # 每隔多少步评估一次 train/val loss
learning_rate = 3e-4          # GPT 标配的小学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 有显卡用显卡
eval_iters = 200              # 评估时平均 200 个 batch，抵消采样噪声
n_embd = 384                  # 每个 token 的特征维度
n_head = 6                    # 注意力头数（head_size = 384/6 = 64）
n_layer = 6                   # Transformer Block 层数
dropout = 0.2                 # 正则化：小数据集上防过拟合
# ------------

torch.manual_seed(1337)

# ---------- 数据：读取 + 字符级分词 ----------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# ---------- 训练/验证切分（90% / 10%） ----------
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 随机抽 batch_size 个窗口起点
    x = torch.stack([data[i:i+block_size] for i in ix])        # 输入 (B, T)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])    # 目标 = 输入右移一位 (B, T)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    # 评估函数：不建计算图；train/val 各抽 eval_iters 个 batch 取平均
    # model.eval() 会关掉 dropout，评估完再切回 model.train()
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # q/k/v 三个无偏置线性投影（n_embd → head_size）：
        # query=我在找什么，key=我有什么，value=我实际携带的信息
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # 下三角掩码（buffer：随模型搬运但不参与训练）

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        # 相似度分数：除以 √head_size 缩放，防止点积随维度变大而过大、softmax 太尖锐
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) 因果掩码：未来位置 → -inf
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs) 按注意力权重加权聚合 value
        return out

# ---------- 多头注意力：多个 Head 并行，再混合回 n_embd ----------
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,nh*hs) 各头结果沿特征维拼接
        out = self.dropout(self.proj(out))                    # proj 把各头信息线性混合回 n_embd
        return out

# ---------- 前馈层 FFWD：每个 token 独立走一遍 384 → 1536 → ReLU → 384 ----------
# 注意力负责 token 之间的「通信」，FFWD 负责单 token 内部的「计算」
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ---------- Transformer Block：通信 + 计算，各带残差；Pre-LN 结构 ----------
# 公式：x = x + sa(ln1(x))；x = x + ffwd(ln2(x))
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # 通信支路（多头注意力）+ 残差
        x = x + self.ffwd(self.ln2(x))    # 计算支路（FFWD）+ 残差
        return x

# ---------- 完整 GPT：词嵌入 + 位置嵌入 → N×Block → ln_f → lm_head ----------
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)      # (65, 384) 每个字符的向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)   # (256, 384) 每个位置的向量
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  # 6 层堆叠
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm（Pre-LN 结构的标配收尾）
        self.lm_head = nn.Linear(n_embd, vocab_size)  # 把 384 维特征投影成 65 个字符的分数

        # better init, not covered in the original GPT video, but important, will cover in followup video
        # 0.02 正态初始化：原版视频没讲，但很重要（后面 nanoGPT 详解）
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

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) 查字符向量
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) 查位置向量（广播加到 batch）
        x = tok_emb + pos_emb # (B,T,C) 词义 + 位置
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)   # 交叉熵要求 (B*T, 65)
            targets = targets.view(B*T)    # 目标拍平成一维
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]   # 只保留最后 block_size 个 token：模型上下文窗口的上限
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) 按概率采样（不是取最大，保留多样性）
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# ---------- 训练：建模型 → AdamW → 循环（定期评估 + 五步曲） ----------
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# 想落盘采样 10000 字符时取消下一行注释（用户实际跑过，产物在 more.txt）
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
