import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr

# ==========================================
# HYPERPARAMETERS
# ==========================================
n_embd = 256 
n_head = 8
n_layer = 6
dropout = 0.25

batch_size = 256
learning_rate = 3e-4
num_epochs = 30

IMG_SIZE = 256
PATCH_SIZE = 16
num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. DATASET (Updated for 3 Classes)
# ==========================================
class StockChartDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Image Loading
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        # Raw Return (Float)
        raw_value = float(self.data_frame.iloc[idx, 1])

        # --- CHANGE 1: 3-CLASS LABELING ---
        # 0: Down ( < -2%), 1: Hold (-2% to +2%), 2: Up ( > +2%)
        if raw_value < -0.02:
            class_id = 0
        elif raw_value > 0.02:
            class_id = 2
        else:
            class_id = 1
            
        target = torch.tensor(class_id, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, target, raw_value

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.values(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
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

class Block(nn.Module):
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

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=n_embd, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        self.position_embedding_table = nn.Embedding(num_patches, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # --- CHANGE 2: OUTPUT NEURONS SET TO 3 ---
        self.lm_head = nn.Linear(n_embd, 3)

    def forward(self, x, targets=None):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        T = x.shape[1]
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))
        x = x + pos_emb
        x = self.blocks(x)
        x, _ = x.max(dim=1)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # --- CHANGE 3: UPDATED CLASS WEIGHTS FOR 3 CLASSES ---
            # Order: [Down, Hold, Up]
            # Since you said they are almost perfect thirds, weights should be near 1.0.
            class_weights = torch.tensor([1.11, 1.00, 1.02]).to(x.device)
            loss = F.cross_entropy(logits, targets, weight=class_weights)

        return logits, loss

# ==========================================
# 3. METRICS (Updated for 3 Classes)
# ==========================================
@torch.no_grad()
def estimate_metrics(model, train_loader, val_loader):
    model.eval()
    out = {}
    
    val_losses = []
    val_correct = 0
    val_total = 0
    
    all_preds_conf = [] 
    all_actual_rets = [] 
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    
    for X, Y, raw_rets in pbar:
        X, Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        val_losses.append(loss.item())
        
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        val_correct += (pred_class == Y).sum().item()
        val_total += Y.size(0)
        
        # --- CHANGE 4: UPDATED RANK IC LOGIC ---
        # We use the probability of Class 2 (UP) as our confidence signal.
        # Alternatively, use (prob_up - prob_down) for a stronger signal.
        prob_signal = (probs[:, 2] - probs[:, 0]).cpu().numpy()
        all_preds_conf.extend(prob_signal)
        all_actual_rets.extend(raw_rets.numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    out['val_loss'] = sum(val_losses) / len(val_losses)
    out['val_acc'] = val_correct / val_total
    
    try:
        rank_ic, _ = spearmanr(all_preds_conf, all_actual_rets)
    except:
        rank_ic = 0.0
    out['val_rank_ic'] = rank_ic
    
    train_losses = []
    for i, (X, Y, _) in enumerate(train_loader):
        if i >= 100: break
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        train_losses.append(loss.item())

    out['train_loss'] = sum(train_losses) / len(train_losses)
    model.train()
    return out

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    print("🚀 Initializing Training (3-Class Classification: Down, Hold, Up)")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    DATA_CSV = '../../datasets/with-dates/data.csv'   
    IMG_DIR = '../../datasets/with-dates/images'
    
    full_dataset = StockChartDataset(csv_file=DATA_CSV, img_dir=IMG_DIR, transform=transform)
    
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = ViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_rank_ic = -1.0 
    batch_step = 0
    
    train_log = open("train_log.csv", "w")
    train_log.write("Epoch,Step,Loss,Acc\n")
    
    print(f"Loaded {len(train_dataset)} samples. Labels: 0(Down), 1(Hold), 2(Up)")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        print(f"\n=== Starting Epoch {epoch} ===")

        for i, (images, targets, _) in enumerate(train_loader):
            batch_step += 1
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(images, targets)
            loss.backward()
            optimizer.step()

            if batch_step % 50 == 0:
                with torch.no_grad():
                    pred_class = torch.argmax(logits, dim=1)
                    acc_score = (pred_class == targets).float().mean().item()
                    print(f"Step {batch_step} | Loss: {loss.item():.4f} | Acc: {acc_score*100:.1f}%")
                    train_log.write(f"{epoch},{batch_step},{loss.item()},{acc_score}\n")
                    train_log.flush()

        metrics = estimate_metrics(model, train_loader, val_loader)
        t_loss = metrics['train_loss']
        v_loss = metrics['val_loss']
        v_acc = metrics['val_acc']
        v_rank_ic = metrics['val_rank_ic']
        
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), 'best_loss_model.pth')
            print(f"💾 Saved Best LOSS Model ({v_loss:.4f})")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), 'best_acc_model.pth')
            print(f"💾 Saved Best ACC Model ({v_acc*100:.2f}%)")
            
        if v_rank_ic > best_val_rank_ic:
            best_val_rank_ic = v_rank_ic
            torch.save(model.state_dict(), 'best_ic_model.pth')
            print(f"🎯 Saved Best RANK IC Model (Corr: {v_rank_ic:.4f})")

        dt = time.time() - t0
        print(f">> Epoch {epoch} | Time: {dt:.1f}s | Val Acc: {v_acc*100:.1f}% | Rank IC: {v_rank_ic:.4f}")

    train_log.close()
    print("\n✅ Training Complete.")

if __name__ == "__main__":
    main()
