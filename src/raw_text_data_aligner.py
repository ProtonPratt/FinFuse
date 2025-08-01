# src/data_processing/raw_text_daily_aligner.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
# Assuming config.py is in src/
from config import RAW_NEWS_DIR, RAW_STOCK_DIR, TICKERS

# Define output directory for this specific alignment step
ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent / "dataset" / "aligned_raw_text_daily"

print(f"Output directory for aligned raw text daily: {ALIGNED_RAW_TEXT_DAILY_DIR}")

def align_raw_news_with_daily_stocks(target_lag_days=1):
    """
    Aligns raw news text with daily stock data.
    News for Day D is aligned with stock data from Day D (and prior)
    and the target is calculated based on stock movement 'target_lag_days' after Day D.

    Args:
        target_lag_days (int): Number of trading days after Day D to calculate the target.
                               e.g., 1 means predict Day D+1 movement using Day D info.
    """
    ALIGNED_RAW_TEXT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers_aligned_data = []

    for ticker in TICKERS:
        print(f"Processing alignment for {ticker} with target lag {target_lag_days} days...")
        try:
            news_path = RAW_NEWS_DIR / f"{ticker}_alpha_news_data.csv"
            stock_path = RAW_STOCK_DIR / f"{ticker}_yahoo_data_0.csv"

            news_df = pd.read_csv(news_path)
            stock_df = pd.read_csv(stock_path, parse_dates=['Date'])
        except FileNotFoundError:
            print(f"Data not found for {ticker} (news: {news_path.exists()}, stock: {stock_path.exists()}). Skipping.")
            continue

        # --- Prepare News Data ---
        # Convert 'published_date' to just date part for daily aggregation
        news_df['date'] = pd.to_datetime(news_df['published_date'], format='%Y%m%dT%H%M%S', errors='coerce').dt.date
        news_df.dropna(subset=['date'], inplace=True)
        news_df['news_text_combined'] = news_df['title'].fillna('') + " <SEP> " + news_df['summary'].fillna('')

        # Aggregate all news texts for a given date
        # Using a list of texts, or joining them with a special separator
        daily_news_agg = news_df.groupby('date')['news_text_combined'].apply(lambda x: ' <NEWS_ITEM_SEP> '.join(x)).reset_index()
        daily_news_agg.rename(columns={'news_text_combined': 'aggregated_news_text_day_D'}, inplace=True)

        # --- Prepare Stock Data ---
        stock_df.rename(columns={'Date': 'datetime_col'}, inplace=True)
        stock_df['date'] = stock_df['datetime_col'].dt.date
        stock_df.sort_values('date', inplace=True)
        stock_df.set_index('date', inplace=True) # Easier for shifting

        # Filter stock data to a relevant range (e.g., around news period)
        min_news_date_for_ticker = daily_news_agg['date'].min()
        max_news_date_for_ticker = daily_news_agg['date'].max()

        if pd.isna(min_news_date_for_ticker) or pd.isna(max_news_date_for_ticker):
            print(f"No valid news dates for {ticker}. Skipping.")
            continue
        
        # Filter stock_df to cover the news period plus buffer for lags and target
        # Find the actual min/max dates in stock_df that bound the news period
        stock_min_date = stock_df.index.min()
        stock_max_date = stock_df.index.max()

        # Ensure min_news_date_for_ticker and max_news_date_for_ticker are within stock_df's range
        effective_min_news_date = max(min_news_date_for_ticker, stock_min_date)
        effective_max_news_date = min(max_news_date_for_ticker, stock_max_date - pd.Timedelta(days=target_lag_days)) # Ensure target can be calculated

        # Select stock data for Day D features
        # We iterate through dates where we have news
        aligned_for_ticker = []
        for current_date in pd.date_range(start=effective_min_news_date, end=effective_max_news_date):
            current_date_obj = current_date.date() # Convert Timestamp to date object for comparison

            if current_date_obj not in stock_df.index: # Skip if it's not a trading day
                continue

            # Get stock data for Day D
            day_D_stock_data = stock_df.loc[current_date_obj]
            
            # Get aggregated news for Day D
            day_D_news = daily_news_agg[daily_news_agg['date'] == current_date_obj]
            aggregated_news = day_D_news['aggregated_news_text_day_D'].iloc[0] if not day_D_news.empty else ""

            # --- Calculate Target ---
            # Target is based on price 'target_lag_days' trading days *after* current_date_obj
            # We need to find the actual trading day corresponding to D + target_lag_days
            
            # Get available trading days after current_date_obj
            future_trading_days = stock_df.loc[stock_df.index > current_date_obj].index
            if len(future_trading_days) < target_lag_days:
                # Not enough future trading days to calculate target
                target_close_price = np.nan
            else:
                target_date = future_trading_days[target_lag_days - 1]
                target_close_price = stock_df.loc[target_date, 'Close']
            
            current_close_price = day_D_stock_data['Close']

            if pd.isna(target_close_price) or pd.isna(current_close_price) or current_close_price == 0:
                target_return = np.nan
                target_direction = np.nan # Or a specific value for 'unknown'
            else:
                target_return = (target_close_price - current_close_price) / current_close_price
                target_direction = 1 if target_return > 0 else 0 # Simple binary direction

            # --- Historical Stock Features for Day D (using data up to and including Day D) ---
            # These are features known at the *end* of Day D, used to predict D+target_lag_days
            # Example: Day D's close, open, high, low, volume, daily_return_Day_D
            # For simplicity, let's just take Close of Day D and return of Day D
            
            # To calculate return on Day D, we need Day D-1's close
            # Find previous trading day
            prior_trading_days = stock_df.loc[stock_df.index < current_date_obj].index
            if not prior_trading_days.empty:
                prev_trading_day = prior_trading_days[-1]
                close_prev_day = stock_df.loc[prev_trading_day, 'Close']
                daily_return_Day_D = (current_close_price - close_prev_day) / close_prev_day if close_prev_day != 0 else 0
            else:
                daily_return_Day_D = np.nan

            aligned_row = {
                'date_D': current_date_obj,
                'ticker': ticker,
                'aggregated_news_text_day_D': aggregated_news,
                'close_price_day_D': current_close_price,
                'open_price_day_D': day_D_stock_data['Open'],
                'high_price_day_D': day_D_stock_data['High'],
                'low_price_day_D': day_D_stock_data['Low'],
                'volume_day_D': day_D_stock_data['Volume'],
                'daily_return_Day_D': daily_return_Day_D, # Return observed on Day D itself
                f'target_close_price_D_plus_{target_lag_days}': target_close_price,
                f'target_return_D_plus_{target_lag_days}': target_return,
                f'target_direction_D_plus_{target_lag_days}': target_direction
            }
            aligned_for_ticker.append(aligned_row)
        
        if aligned_for_ticker:
            ticker_df = pd.DataFrame(aligned_for_ticker)
            # Drop rows where target could not be computed or essential data is missing
            ticker_df.dropna(subset=[f'target_direction_D_plus_{target_lag_days}'], inplace=True)
            all_tickers_aligned_data.append(ticker_df)

    if not all_tickers_aligned_data:
        print("No data was aligned for any ticker.")
        return

    final_aligned_df = pd.concat(all_tickers_aligned_data, ignore_index=True)
    
    output_filename = f"all_tickers_raw_text_aligned_daily_lag{target_lag_days}.csv"
    output_path = ALIGNED_RAW_TEXT_DAILY_DIR / output_filename
    final_aligned_df.to_csv(output_path, index=False)
    print(f"Saved aligned raw text daily data to {output_path}. Shape: {final_aligned_df.shape}")
    if not final_aligned_df.empty:
        print(final_aligned_df.head())
        print(final_aligned_df[f'target_direction_D_plus_{target_lag_days}'].value_counts(normalize=True, dropna=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Align raw news with daily stock data.")
    parser.add_argument(
        '--target_lag_days',
        type=int,
        default=1,
        help="Number of trading days after Day D to calculate the target stock movement. Default is 1 (for next trading day)."
    )
    args = parser.parse_args()

    # Ensure TICKERS is available (e.g., imported from config)
    # from src.config import TICKERS (if config.py is in src/)
    # Or define it here if this script is standalone for now
    # TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA", "NKE"] 

    align_raw_news_with_daily_stocks(target_lag_days=args.target_lag_days)