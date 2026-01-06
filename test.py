import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import os
import sys

# --- CONFIGURATION (MUST MATCH V2 TRAINING) ---
MODEL_PATH = '../../chart-vision/models/best_acc_model.pth'  # Path to your TIME-SPLIT model
DATA_CSV = '../../datasets/with-dates/data.csv'   
IMG_DIR = '../../datasets/with-dates/images'

# Model Hyperparameters
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.25
IMG_SIZE = 256
PATCH_SIZE = 16
num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

BATCH_SIZE = 256 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"⚙️  Config: {IMG_SIZE}x{IMG_SIZE} images | Patch Size {PATCH_SIZE} | Device: {DEVICE}")

# ==========================================
# 1. MODEL ARCHITECTURE (V2: 3-Class)
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
        self.patch_embed = nn.Conv2d(in_channels=3,
                                     out_channels=n_embd,
                                     kernel_size=PATCH_SIZE,
                                     stride=PATCH_SIZE)
        self.position_embedding_table = nn.Embedding(num_patches, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # --- UPDATE: 3 Output Neurons [Down, Hold, Up] ---
        self.lm_head = nn.Linear(n_embd, 3)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.position_embedding_table(torch.arange(x.shape[1], device=x.device))
        x = self.blocks(x); x, _ = x.max(dim=1)
        return self.lm_head(x)

# ==========================================
# 2. DATASET & VALIDATION SPLIT (Time-Split Logic)
# ==========================================

class StockChartDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        # We accept the split DataFrame directly
        self.data_frame = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        
        # --- 3-CLASS LOGIC ---
        raw_value = float(self.data_frame.iloc[idx, 1])
        
        if raw_value < -0.02:
            class_id = 0 # Down
        elif raw_value > 0.02:
            class_id = 2 # Up
        else:
            class_id = 1 # Hold
        # if raw_value < -0.03:
        #     class_id = 0  # Down 
        # elif raw_value > 0.01:
        #     class_id = 2  # Up 
        # else:
        #     class_id = 1  # Hold
    
        target = torch.tensor(class_id, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)

        return image, target

def get_validation_loader():
    print("Preparing Validation Data (Time Split)...")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. LOAD AND SORT
    print("📅 Loading and Sorting Data by Date...")
    df = pd.read_csv(DATA_CSV, header=0, names=['filename', 'return', 'ticker', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. CALCULATE SPLIT (Same as training)
    total_len = len(df)
    train_size = int(0.8 * total_len)
    
    # 3. APPLY EMBARGO (90 Days)
    split_date = df.iloc[train_size]['date']
    embargo_date = split_date + pd.Timedelta(days=90)
    
    # We only care about the Validation DataFrame here
    val_df = df[df['date'] > embargo_date].copy()
    
    print(f"Full Dataset:   {total_len}")
    print(f"Train Cutoff:   {split_date.date()}")
    print(f"Val Start:      {embargo_date.date()}")
    print(f"Testing on:     {len(val_df)} samples")
    
    val_dataset = StockChartDataset(val_df, img_dir=IMG_DIR, transform=transform)
    
    return DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================================
# 3. AUDIT EXECUTION
# ==========================================

def main():
    # Load Model
    print(f"\nLoading Model Weights: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found!")
        return

    model = ViT().to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
        
    model.eval()

    # Load Data
    val_loader = get_validation_loader()

    # Run Inference
    all_preds = []
    all_targets = []
    all_probs = []

    print("\nRunning Inference on Validation Set...")
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(DEVICE)
            
            # Forward pass
            logits = model(x)
            
            # Get Probabilities
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            all_probs.extend(max_probs.cpu().numpy())

    # ==========================================
    # 4. REPORTING (3-Class)
    # ==========================================
    
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. Basic Accuracy
    acc = accuracy_score(all_targets, all_preds)
    
    print("\n" + "="*40)
    print(f"FINAL VALIDATION ACCURACY: {acc*100:.2f}%")
    print("="*40)
    
    # 2. Classification Report
    target_names = ['Down (0)', 'Hold (1)', 'Up (2)']
    print("\n--- DETAILED CLASS PERFORMANCE ---")
    print(classification_report(all_targets, all_preds, target_names=target_names))

    # 3. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    print("\n--- CONFUSION MATRIX (Truth x Pred) ---")
    print(f"{'':>10} | {'Pred Down':>10} | {'Pred Hold':>10} | {'Pred Up':>10}")
    print("-" * 50)
    print(f"{'True Down':>10} | {cm[0,0]:>10} | {cm[0,1]:>10} | {cm[0,2]:>10}")
    print(f"{'True Hold':>10} | {cm[1,0]:>10} | {cm[1,1]:>10} | {cm[1,2]:>10}")
    print(f"{'True Up':>10}   | {cm[2,0]:>10} | {cm[2,1]:>10} | {cm[2,2]:>10}")
    
    # Insight logic
    total_longs = cm[:, 2].sum()
    total_shorts = cm[:, 0].sum()
    
    print("\n--- BEHAVIOR ANALYSIS ---")
    print(f"Total Short Signals: {total_shorts}")
    print(f"Total Long Signals:  {total_longs}")
    
    if total_shorts > 100 and total_longs > 100:
         print("✅ BALANCED: Model is trading both sides.")
    elif total_longs == 0:
         print("⚠️ WARNING: Model is not taking any Longs.")
    
    # 4. Confidence Analysis
    print("\n" + "="*40)
    print("CONFIDENCE ANALYSIS (Accuracy by Certainty)")
    print("="*40)
    
    df = pd.DataFrame({'target': all_targets, 'pred': all_preds, 'prob': all_probs})
    df['correct'] = df['target'] == df['pred']
    
    # Bins for confidence
    bins = [0.33, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    labels = ['33-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80%+']
    
    df['conf_bin'] = pd.cut(df['prob'], bins=bins, labels=labels, right=False)
    
    grouped = df.groupby('conf_bin', observed=False)['correct'].agg(['mean', 'count'])
    grouped = grouped.rename(columns={'mean': 'Accuracy', 'count': 'Trades'})
    
    # Format accuracy as percentage
    grouped['Accuracy'] = grouped['Accuracy'].mul(100).round(2).astype(str) + '%'
    
    print(grouped)

if __name__ == "__main__":
    main()
