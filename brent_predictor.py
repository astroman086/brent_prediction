import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import nltk
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- NLTK Secure Setup ---
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class BrentCrudePredictorTask:
    def __init__(self, log_file="brent_predictions_log.csv"):
        self.log_file = log_file
        self.short_term_horizons = {
            '5m': 1, '30m': 6, '1h': 12, '2h': 24, 
            '5h': 60, '12h': 144, '24h': 288
        }
        self.long_term_horizons = {
            '5d': 5, '10d': 10, '15d': 15, 
            '30d': 21, '6months': 126, '1y': 252
        }
        self.scaler = StandardScaler()

    def fetch_market_data(self):
        df_hourly = yf.download("BZ=F", period="60d", interval="1h", progress=False)
        if isinstance(df_hourly.columns, pd.MultiIndex):
            df_hourly.columns = df_hourly.columns.droplevel(1)
            
        df_daily = yf.download("BZ=F", period="max", interval="1d", progress=False)
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.droplevel(1)
            
        return df_hourly.dropna(), df_daily.dropna()

    def fetch_news_sentiment(self):
        analyzer = SentimentIntensityAnalyzer()
        rss_urls = [
            "https://news.google.com/rss/search?q=brent+crude+oil+price+opec&hl=en-US&gl=US&ceid=US:en",
            "https://www.oilprice.com/rss/main"
        ]
        sentiments = []
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    sentiments.append(analyzer.polarity_scores(entry.title)['compound'])
            except Exception:
                pass
        return float(sum(sentiments) / len(sentiments)) if sentiments else 0.0

    def engineer_features(self, df):
        df['Returns'] = df['Close'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = (df['High'] - df['Low']).rolling(window=10).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-6))))
        
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['Macro_Trend'] = df['Close'] - df['MA_50']
        df['Sentiment'] = self.fetch_news_sentiment()

        return df.dropna()

    def run(self):
        print(f"--- EXECUTING PREDICTION RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        df_hourly, df_daily = self.fetch_market_data()
        processed_st = self.engineer_features(df_hourly)
        processed_lt = self.engineer_features(df_daily)
        
        current_price = float(processed_st['Close'].iloc[-1])
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        st_features = ['Close', 'Returns', 'MA_5', 'MA_20', 'Volatility', 'RSI', 'Sentiment']
        lt_features = ['Close', 'MA_20', 'MA_50', 'Macro_Trend', 'Sentiment']
        results = {}
        
        # Short-Term
        X_st = self.scaler.fit_transform(processed_st[st_features].values)
        for horizon, steps in self.short_term_horizons.items():
            processed_st[f'target_{horizon}'] = processed_st['Close'].shift(-min(steps, len(processed_st)-1))
            train_df = processed_st.dropna(subset=[f'target_{horizon}'])
            model = RandomForestRegressor(n_estimators=60, random_state=42, max_depth=5)
            model.fit(self.scaler.transform(train_df[st_features].values), train_df[f'target_{horizon}'].values)
            pred = model.predict(X_st[-1].reshape(1, -1))[0]
            results[horizon] = (round(float(pred), 2), round(float(((pred - current_price) / current_price) * 100), 2))

        # Long-Term
        X_lt = self.scaler.fit_transform(processed_lt[lt_features].values)
        for horizon, steps in self.long_term_horizons.items():
            processed_lt[f'target_{horizon}'] = processed_lt['Close'].shift(-min(steps, len(processed_lt)-1))
            train_df = processed_lt.dropna(subset=[f'target_{horizon}'])
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
            model.fit(self.scaler.transform(train_df[lt_features].values), train_df[f'target_{horizon}'].values)
            pred = model.predict(X_lt[-1].reshape(1, -1))[0]
            results[horizon] = (round(float(pred), 2), round(float(((pred - current_price) / current_price) * 100), 2))

        rows = []
        for horizon, (pred_price, pct) in results.items():
            rows.append({
                "Log_Timestamp": current_time,
                "Base_Price": current_price,
                "Horizon": horizon,
                "Predicted_Price": pred_price,
                "Predicted_Change_Pct": pct,
                "Actual_Price": np.nan,
                "Error_Pct": np.nan
            })
        
        df_new_log = pd.DataFrame(rows)
        file_exists = os.path.exists(self.log_file)
        
        if file_exists:
            df_new_log.to_csv(self.log_file, mode='a', header=False, index=False)
        else:
            df_new_log.to_csv(self.log_file, mode='w', header=True, index=False)
            
        with open(self.log_file, 'a') as f:
            f.write('\n')

        print(f"[+] Saved new predictions to {self.log_file}")


if __name__ == "__main__":
    BrentCrudePredictorTask().run()
