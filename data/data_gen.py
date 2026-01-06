import yfinance as yf
import mplfinance as mpf
import pandas as pd
import os
import requests
import time
import gc
import matplotlib.pyplot as plt 
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
IMG_DIR = 'with-dates/images'
CSV_PATH = 'with-dates/data.csv'
CHECKPOINT_FILE = 'with-dates/completed_tickers.txt'

# Window Settings
WINDOW_SIZE = 60
PREDICT_DAYS = 7
GAP_SIZE = 20          # Skip 20 days between samples to reduce correlation
STEP_SIZE = WINDOW_SIZE + GAP_SIZE 

# Image Settings
IMG_SIZE_INCHES = 5.12 # 5.12 * 100 dpi = 512 pixels (High Res for Downscaling)
DPI = 100

# Filters
MIN_PRICE = 5.0        # Skip stocks under $5 (Penny stocks are noisy)
MIN_LIQUIDITY = 500000 # Min Avg Dollar Volume (Price * Vol) to ensure liquidity

BATCH_SIZE = 10        # Save progress every 10 tickers

# ==========================================
# SETUP
# ==========================================
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def load_checkpoint():
    """Returns a set of tickers that have already been processed."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r') as f:
        # Read lines, strip whitespace, add to set
        return set(line.strip() for line in f)

def save_checkpoint(tickers_list):
    """Appends a list of completed tickers to the checkpoint file."""
    with open(CHECKPOINT_FILE, 'a') as f:
        for t in tickers_list:
            f.write(f"{t}\n")

def get_all_tickers():
    """Fetches the master list of US tickers from the SEC official JSON."""
    try:
        print("🌐 Fetching master ticker list from SEC...")
        
        # The SEC specifically requires a User-Agent with an email or distinct ID
        headers = {
            'User-Agent': 'ChartVisionAudit/1.0 (audit@example.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        url = "https://www.sec.gov/files/company_tickers.json"
        
        # 1. Use requests to get the data (bypasses the 403 block)
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for errors
        
        # 2. Parse the JSON
        data = response.json()
        
        # Extract just the ticker symbols
        tickers = [entry['ticker'] for entry in data.values()]
        
        # 3. Clean Tickers
        # Convert dots to hyphens (e.g., BRK.B -> BRK-B) for yfinance compatibility
        tickers = [str(t).replace('.', '-') for t in tickers]
        
        # Sort and remove duplicates
        tickers = sorted(list(set(tickers)))
        
        print(f"✅ Success! Found {len(tickers)} active tickers from SEC.")
        return tickers

    except Exception as e:
        print(f"❌ Error fetching from SEC ({e}).")
        print("   Falling back to S&P 500...")
        # Fallback to Wikipedia if SEC fails
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return [t.replace('.', '-') for t in table[0]['Symbol'].tolist()]
        except:
            return ['AAPL', 'MSFT', 'TSLA', 'NVDA'] # Emergency fallback

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # 1. Init
    TICKERS = get_all_tickers()
    completed_tickers = load_checkpoint()
    
    # Filter out already done
    tickers_to_do = [t for t in TICKERS if t not in completed_tickers]
    print(f"resume: Skipping {len(completed_tickers)} tickers. {len(tickers_to_do)} remaining.")

    dataset_records = []
    current_batch_tickers = []
    img_counter = len(os.listdir(IMG_DIR)) # Start counter based on existing files

    # 2. Loop
    pbar = tqdm(tickers_to_do, desc="Generating Dataset")
    
    for ticker in pbar:
        try:
            # --- A. DOWNLOAD DATA ---
            # 'period="max"' ensures we get 10-20 years of data if available
            df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=True)
            
            if len(df) < 300: # Need at least ~1 year of data for 200 SMA + Window
                current_batch_tickers.append(ticker)
                continue

            # Handle MultiIndex columns (yfinance update quirk)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

            # --- B. PRE-CALCULATE INDICATORS (THE STACK) ---
            # 1. The Trinity
            df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            # 2. Bollinger Bands
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD20'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['MA20'] + (2 * df['STD20'])
            df['BB_Lower'] = df['MA20'] - (2 * df['STD20'])
            
            # 3. Dollar Volume (For Liquidity Filter)
            df['DollarVol'] = df['Close'] * df['Volume']
            df['AvgDollarVol'] = df['DollarVol'].rolling(window=WINDOW_SIZE).mean()

            # --- C. SLIDING WINDOW ---
            # Start at 200 so MA200 is valid
            valid_rows = 0
            
            for i in range(200, len(df) - WINDOW_SIZE - PREDICT_DAYS, STEP_SIZE):
                window = df.iloc[i : i+WINDOW_SIZE]
                
                # --- D. QUALITY FILTERS ---
                current_price = window['Close'].iloc[-1]
                avg_liq = window['AvgDollarVol'].iloc[-1]
                
                # Filter 1: Penny Stocks
                if current_price < MIN_PRICE:
                    continue
                
                # Filter 2: Illiquid Stocks (Hard to trade, noisy charts)
                if avg_liq < MIN_LIQUIDITY:
                    continue
                
                # Filter 3: Empty Volume (Data error)
                if window["Volume"].sum() == 0:
                    continue

                # --- E. GENERATION ---
                future_price = df['Close'].iloc[i + WINDOW_SIZE + PREDICT_DAYS - 1]
                pct_change = (future_price - current_price) / current_price
                
                fname = f"img_{ticker}_{img_counter:07d}.png" # Include ticker in filename for debug
                save_path = os.path.join(IMG_DIR, fname)
                
                try:
                    # Configure Custom Plots
                    ap = [
                        # Bollinger (Cyan, Thin, Behind)
                        mpf.make_addplot(window['BB_Upper'], color='cyan', width=0.8),
                        mpf.make_addplot(window['BB_Lower'], color='cyan', width=0.8),
                        # Trinity
                        mpf.make_addplot(window['EMA9'], color='orange', width=1.0),
                        mpf.make_addplot(window['MA50'], color='green', width=1.5),
                        mpf.make_addplot(window['MA200'], color='blue', width=2.0)
                    ]

                    mpf.plot(window, 
                             type='candle', 
                             style='charles', 
                             volume=True, 
                             axisoff=True, 
                             addplot=ap,
                             panel_ratios=(4,1),
                             savefig=dict(fname=save_path, bbox_inches='tight', pad_inches=0),
                             figsize=(IMG_SIZE_INCHES, IMG_SIZE_INCHES) # 512x512
                    )
                    plt.close('all') # Critical for memory
                    
                    # Inside your loop, right before appending to dataset_records
                    last_date = window.index[-1] # Assuming dataframe index is Datetime (yfinance default)

                    dataset_records.append({
                        'filename': fname,
                        'target': float(pct_change),
                        'ticker': ticker, # Useful for later analysis
                        'date': str(last_date.date()) # Add this!
                    })
                    img_counter += 1
                    valid_rows += 1
                    
                except Exception:
                    plt.close('all')
                    continue

            # Mark ticker as done locally
            current_batch_tickers.append(ticker)

            # --- F. BATCH SAVE & CHECKPOINT ---
            if len(current_batch_tickers) >= BATCH_SIZE:
                # 1. Save CSV Data
                if dataset_records:
                    df_batch = pd.DataFrame(dataset_records)
                    file_exists = os.path.isfile(CSV_PATH)
                    df_batch.to_csv(CSV_PATH, mode='a', index=False, header=not file_exists)
                    dataset_records = [] # Clear memory
                
                # 2. Update Checkpoint File
                save_checkpoint(current_batch_tickers)
                current_batch_tickers = [] # Clear list
                
                # 3. Garbage Collection
                gc.collect()

        except Exception as e:
            # If a specific ticker fails (API error, etc), log it and move on
            # We do NOT add it to checkpoint so we can try it again later if needed
            print(f"\n⚠️ Skipped {ticker}: {e}")
            time.sleep(1) # Cooldown
            continue

    # Final cleanup for last batch
    if dataset_records:
        df_batch = pd.DataFrame(dataset_records)
        df_batch.to_csv(CSV_PATH, mode='a', index=False, header=not os.path.isfile(CSV_PATH))
    
    if current_batch_tickers:
        save_checkpoint(current_batch_tickers)

    print(f"\n🎉 DONE! Total Images Generated: {img_counter}")

if __name__ == "__main__":
    main()