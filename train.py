import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from scheduler import CosineScheduler
from tokenizer import tokenizer
from BigramGFN import BigramLanguageModel
torch.manual_seed(42)


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


with open("/kaggle/input/datapoem/fulldata2.txt",'r',encoding='utf-8') as f:
    text = f.read()


tokenizer = tokenizer(text,128)
text, vocab_size = tokenizer.initialisation()


decoding_table, encoding_table  = tokenizer.train_BPE(10000)

data=torch.tensor(tokenizer.encode(text,encoding_table), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:] 

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])

    return x.to(device),y.to(device)


@torch.no_grad() #no need for gradients
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)


            if (X >= vocab_size).any() or (Y >= vocab_size).any(): #debug
                print("Error: Found indices out of bounds in the data.")
                print("X:", X)
                print("Y:", Y)
            logits, loss = model(X,Y)
            losses[k]= loss.item()

        out[split] = losses.mean()
    model.train()
    return out


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
scheduler = CosineScheduler(optimizer, 0, max_iters, 1e-6, learning_rate)

scaler = GradScaler()
for iter in range(max_iters):
    if iter % eval_interval ==0:
        current_lr = optimizer.param_groups[0]['lr']
        losses = estimate_loss()
        print(f"step{iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, current lr {current_lr}")
    if iter % 1000 == 0 and iter > 100:
        file_name = "bigram_language_model_state_iter"+str(iter)+".pth"
        torch.save(model.state_dict(), file_name)
    
    xb,yb= get_batch('train')
    with autocast(device_type='cuda'):  # Enable mixed precision for forward pass
          logits, loss = m(xb, yb)
   
    scaler.scale(loss).backward()  
    
    if (iter + 1) % accumulation_steps == 0:
        
        scaler.step(optimizer)  
        scaler.update() 
        optimizer.zero_grad()
    scheduler.step()




torch.save(model.state_dict(), "bigram_language_model_final.pth")

