import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = 'datasets/with-dates/data.csv' # Adjust if your path is different
THRESHOLDS_TO_TEST = [0.005, 0.01, 0.02] # 0.5%, 1.0%, 2.0%

def calculate_weights(counts):
    """
    Calculates inverse class weights.
    Formula: Total_Samples / (Num_Classes * Class_Count)
    This makes the model pay more attention to rare classes.
    """
    total = sum(counts)
    n_classes = len(counts)
    weights = []
    
    for count in counts:
        if count > 0:
            w = total / (n_classes * count)
        else:
            w = 0.0
        weights.append(w)
        
    # Normalize so the smallest weight is roughly 1.0 (optional, but easier to read)
    min_w = min([w for w in weights if w > 0])
    normalized_weights = [w / min_w for w in weights]
    
    return normalized_weights

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found.")
        return

    print(f"📂 Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Ensure target is float
    # Assuming 'target' is the 2nd column (index 1) or named 'target'
    if 'target' in df.columns:
        targets = df['target']
    else:
        # Fallback to 2nd column if no header
        targets = df.iloc[:, 1]

    print(f"📊 Total Samples: {len(df)}")
    print("-" * 60)

    for thresh in THRESHOLDS_TO_TEST:
        print(f"\n🔍 TESTING THRESHOLD: +/- {thresh*100:.1f}%")
        
        # Categorize
        # Class 0: DOWN ( < -thresh )
        # Class 1: NEUTRAL ( between -thresh and +thresh )
        # Class 2: UP ( > +thresh )
        
        downs = targets[targets < -thresh]
        ups = targets[targets > thresh]
        neutrals = targets[(targets >= -thresh) & (targets <= thresh)]
        
        c0 = len(downs)
        c1 = len(neutrals)
        c2 = len(ups)
        
        pct0 = c0 / len(df) * 100
        pct1 = c1 / len(df) * 100
        pct2 = c2 / len(df) * 100
        
        print(f"   🔴 DOWN (Class 0):    {c0:5d} ({pct0:5.1f}%)")
        print(f"   ⚪ HOLD (Class 1):    {c1:5d} ({pct1:5.1f}%)")
        print(f"   🟢 UP   (Class 2):    {c2:5d} ({pct2:5.1f}%)")
        
        # Calculate Weights
        counts = [c0, c1, c2]
        weights = calculate_weights(counts)
        
        print(f"   ⚖️  Suggested Weights: {weights}")
        print(f"       torch.tensor([{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}])")

        # Recommendation Logic
        if pct1 > 70:
            print("   ⚠️  WARNING: Too much Neutral data. The model might just learn to 'Hold' forever.")
            print("       -> Try lowering the threshold.")
        elif pct0 < 10 or pct2 < 10:
            print("   ⚠️  WARNING: Up/Down classes are too rare. The model will struggle to find them.")
            print("       -> Try lowering the threshold or getting more volatile stocks.")
        else:
            print("   ✅ GOOD BALANCE: This threshold looks healthy.")

if __name__ == "__main__":
    main()