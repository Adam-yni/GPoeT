import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)


#Setup used for training
batch_size = 16
accumulation_steps = 6
block_size = 256
max_iters= 10000
eval_interval = 1000
learning_rate =5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"the device is {device}")
eval_iters= 1000
n_embd = 500
n_head = 20
n_layer= 16
dropout = 0.2





class Head(nn.Module):
    """
        one head attention
    """

    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))

    def apply_rotary_embedding(self,q, k, theta=1e-4): # RoPE

        B, T, C = q.shape
        theta=torch.tensor([theta**(2 * (i // 2) / C) for i in range(C // 2)], device=q.device)
        half_dim = C // 2  

        pos = torch.arange(T, device=q.device).unsqueeze(1)  # (T, 1)
        theta = theta[:half_dim].unsqueeze(0) 

        cos_pos = torch.cos(pos * theta)  # (T, half_dim)
        sin_pos = torch.sin(pos * theta)  # (T, half_dim)
        
        q1, q2 = q[:, :, :half_dim], q[:, :, half_dim:]
        k1, k2 = k[:, :, :half_dim], k[:, :, half_dim:]
        
        q_rotated = torch.cat([q1 * cos_pos - q2 * sin_pos, q1 * sin_pos + q2 * cos_pos], dim=-1)
        k_rotated = torch.cat([k1 * cos_pos - k2 * sin_pos, k1 * sin_pos + k2 * cos_pos], dim=-1)
        
        return q_rotated, k_rotated

    def forward(self,x):
        B,T,C = x.shape
        k=self.key(x)
        q=self.query(x)

        q,k = self.apply_rotary_embedding(q,k) #ROPE

        wei = q @ k.transpose(-2,-1)* C**-.5 # (B,T,16) @ (B,16,T) --> (B,T,T)

        tril= self.tril[:T,:T]

        wei = wei.masked_fill(tril==0, float('-inf')) #decoder
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)


        v = self.value(x)
        out = wei @ v
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self,head_size,num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout=nn.Dropout(dropout) #help to avoid overfitting

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), #projection 
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.network(x)
    

class Block(nn.Module):
    def __init__(self,n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attentions = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self,x):
        x = x+self.self_attentions(self.ln1(x)) #residual connections+layer norm
        x = x+ self.ffwd(self.ln2(x)) #residual connections+layer norm
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.ln_final = nn.LayerNorm(n_embd)


        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])


    def forward(self,idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x =self.blocks(x) # (B,T,C)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        

        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    
    def gfn_generate(self, idx, max_new_tokens,temperature=1.0): #useful for GFlowNet finetuning 
        probs_tokens=[]
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:] #for position embedding to work

            logits, loss = self(idx_cond)

            logits = logits[:,-1,:] #(B,C)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
                

            probs_tokens.append(probs[0, idx_next]) #prob of the chosen token
            idx=torch.cat((idx, idx_next), dim=1)
 
            if idx_next == 77 and idx[0][-2] != 77: # token 77 = \n

                break

        probs_end_token = probs[0, [77]] 
        return idx, probs_tokens, probs_end_token 

    
    def generate(self, idx, max_new_tokens,temperature=1.0, greedy=False):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:] #for position embedding to work

            logits, loss = self(idx_cond)

            logits = logits[:,-1,:] #(B,C)

            if temperature != 1.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            else:
                probs = F.softmax(logits, dim=-1)
                if greedy:
                    idx_next = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)

            idx=torch.cat((idx, idx_next), dim=1)

        return idx
