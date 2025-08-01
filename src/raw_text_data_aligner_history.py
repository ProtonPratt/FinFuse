# src/data_processing/raw_text_daily_aligner.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from config import RAW_NEWS_DIR, RAW_STOCK_DIR, TICKERS # Assuming config.py is in src/

ALIGNED_RAW_TEXT_DAILY_DIR = Path(__file__).resolve().parent.parent / "dataset" / "aligned_raw_text_daily_history"
print(f"Output directory for aligned raw text daily with history: {ALIGNED_RAW_TEXT_DAILY_DIR}")

def align_raw_news_with_daily_stocks(target_lag_days=1, history_lags_days=[1, 2, 3, 5]):
    """
    Aligns raw news text with daily stock data, including lagged historical stock features.
    """
    ALIGNED_RAW_TEXT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers_aligned_data = []

    for ticker in TICKERS:
        print(f"Processing alignment for {ticker} with target lag {target_lag_days} days, history lags: {history_lags_days}...")
        try:
            news_path = RAW_NEWS_DIR / f"{ticker}_alpha_news_data.csv"
            stock_path = RAW_STOCK_DIR / f"{ticker}_yahoo_data_0.csv"
            news_df = pd.read_csv(news_path)
            stock_df = pd.read_csv(stock_path, parse_dates=['Date'])
        except FileNotFoundError:
            print(f"Data not found for {ticker}. Skipping.")
            continue

        news_df['date'] = pd.to_datetime(news_df['published_date'], format='%Y%m%dT%H%M%S', errors='coerce').dt.date
        news_df.dropna(subset=['date'], inplace=True)
        news_df['news_text_combined'] = news_df['title'].fillna('') + " <SEP> " + news_df['summary'].fillna('')
        daily_news_agg = news_df.groupby('date')['news_text_combined'].apply(lambda x: ' <NEWS_ITEM_SEP> '.join(x)).reset_index()
        daily_news_agg.rename(columns={'news_text_combined': 'aggregated_news_text_day_D'}, inplace=True)

        stock_df.rename(columns={'Date': 'datetime_col'}, inplace=True)
        stock_df['date'] = stock_df['datetime_col'].dt.date
        stock_df.sort_values('date', inplace=True)
        
        # Pre-calculate daily returns for entire stock_df for easier lookup
        stock_df['daily_return'] = stock_df['Close'].pct_change() # This is return ending on current row's date
        stock_df.set_index('date', inplace=True)

        min_news_date_for_ticker = daily_news_agg['date'].min()
        max_news_date_for_ticker = daily_news_agg['date'].max()
        if pd.isna(min_news_date_for_ticker) or pd.isna(max_news_date_for_ticker):
            print(f"No valid news dates for {ticker}. Skipping.")
            continue
        
        stock_min_date = stock_df.index.min()
        stock_max_date = stock_df.index.max()
        effective_min_news_date = max(min_news_date_for_ticker, stock_min_date)
        effective_max_news_date = min(max_news_date_for_ticker, stock_max_date - pd.Timedelta(days=target_lag_days))

        aligned_for_ticker = []
        # Ensure we iterate over dates that have news
        dates_with_news = sorted(daily_news_agg[
            (daily_news_agg['date'] >= effective_min_news_date) &
            (daily_news_agg['date'] <= effective_max_news_date)
        ]['date'].unique())


        for current_date_obj in tqdm(dates_with_news, desc=f"Aligning {ticker}"):
            if current_date_obj not in stock_df.index:
                continue # Not a trading day with stock data

            day_D_stock_data = stock_df.loc[current_date_obj]
            day_D_news = daily_news_agg[daily_news_agg['date'] == current_date_obj]
            aggregated_news = day_D_news['aggregated_news_text_day_D'].iloc[0] if not day_D_news.empty else ""

            # --- Target Calculation ---
            future_trading_days = stock_df.loc[stock_df.index > current_date_obj].index
            if len(future_trading_days) < target_lag_days:
                target_close_price = np.nan
            else:
                target_date = future_trading_days[target_lag_days - 1]
                target_close_price = stock_df.loc[target_date, 'Close']
            
            current_close_price = day_D_stock_data['Close']
            if pd.isna(target_close_price) or pd.isna(current_close_price) or current_close_price == 0:
                target_return, target_direction = np.nan, np.nan
            else:
                target_return = (target_close_price - current_close_price) / current_close_price
                target_direction = 1 if target_return > 0 else 0

            # --- Current Day D Stock Features ---
            aligned_row = {
                'date_D': current_date_obj,
                'ticker': ticker,
                'aggregated_news_text_day_D': aggregated_news,
                'close_price_day_D': current_close_price,
                'open_price_day_D': day_D_stock_data['Open'],
                'high_price_day_D': day_D_stock_data['High'],
                'low_price_day_D': day_D_stock_data['Low'],
                'volume_day_D': day_D_stock_data['Volume'],
                'daily_return_Day_D': day_D_stock_data['daily_return'], # Return for Day D
                f'target_close_price_D_plus_{target_lag_days}': target_close_price,
                f'target_return_D_plus_{target_lag_days}': target_return,
                f'target_direction_D_plus_{target_lag_days}': target_direction
            }

            # --- Lagged Historical Stock Features (from Day D-1, D-2, etc.) ---
            current_date_location_in_stock_df_index = stock_df.index.get_loc(current_date_obj)
            for lag in history_lags_days:
                past_date_location = current_date_location_in_stock_df_index - lag
                if past_date_location >= 0:
                    past_date_data = stock_df.iloc[past_date_location]
                    aligned_row[f'close_price_day_D_minus_{lag}'] = past_date_data['Close']
                    aligned_row[f'daily_return_Day_D_minus_{lag}'] = past_date_data['daily_return']
                    aligned_row[f'volume_Day_D_minus_{lag}'] = past_date_data['Volume']
                else:
                    aligned_row[f'close_price_day_D_minus_{lag}'] = np.nan
                    aligned_row[f'daily_return_Day_D_minus_{lag}'] = np.nan
                    aligned_row[f'volume_Day_D_minus_{lag}'] = np.nan
            
            aligned_for_ticker.append(aligned_row)
        
        if aligned_for_ticker:
            ticker_df = pd.DataFrame(aligned_for_ticker)
            ticker_df.dropna(subset=[f'target_direction_D_plus_{target_lag_days}'], inplace=True)
            # Also drop rows if all historical lagged features are NaN (optional, but good for model)
            # example_lagged_col = f'close_price_day_D_minus_{history_lags_days[0]}'
            # ticker_df.dropna(subset=[example_lagged_col], inplace=True)
            all_tickers_aligned_data.append(ticker_df)

    if not all_tickers_aligned_data:
        print("No data was aligned for any ticker.")
        return

    final_aligned_df = pd.concat(all_tickers_aligned_data, ignore_index=True)
    history_lags_str = "_".join(map(str, history_lags_days))
    output_filename = f"all_tickers_raw_text_aligned_daily_lag{target_lag_days}_hist{history_lags_str}.csv"
    output_path = ALIGNED_RAW_TEXT_DAILY_DIR / output_filename
    final_aligned_df.to_csv(output_path, index=False)
    print(f"Saved aligned raw text daily data to {output_path}. Shape: {final_aligned_df.shape}")
    if not final_aligned_df.empty:
        print(final_aligned_df.head())
        print(final_aligned_df[f'target_direction_D_plus_{target_lag_days}'].value_counts(normalize=True, dropna=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Align raw news with daily stock data, including historical lags.")
    parser.add_argument(
        '--target_lag_days', type=int, default=1,
        help="Number of trading days after Day D to calculate the target stock movement."
    )
    parser.add_argument(
        '--history_lags', type=str, default="1,2,3,5",
        help="Comma-separated list of past days for historical stock features (e.g., '1,2,3')."
    )
    args = parser.parse_args()
    history_lags_list = [int(lag.strip()) for lag in args.history_lags.split(',')]

    align_raw_news_with_daily_stocks(
        target_lag_days=args.target_lag_days,
        history_lags_days=history_lags_list
    )