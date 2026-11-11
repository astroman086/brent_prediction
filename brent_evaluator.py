import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


class BrentEvaluatorTask:
    def __init__(self, log_file="brent_predictions_log.csv"):
        self.log_file = log_file

    def run(self):
        if not os.path.exists(self.log_file):
            print(f"[-] No log file found.")
            return

        df_log = pd.read_csv(self.log_file).dropna(how='all')
        if df_log.empty:
            print(f"[-] Log file is empty.")
            return

        print(f"--- EVALUATION RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        try:
            latest_market = yf.download("BZ=F", period="5d", interval="1h", progress=False)
            if isinstance(latest_market.columns, pd.MultiIndex):
                latest_market.columns = latest_market.columns.droplevel(1)
            current_live_price = float(latest_market['Close'].iloc[-1])
        except Exception as e:
            print(f"[-] Error fetching market quotes: {e}")
            return

        delta_map = {
            '5m': timedelta(minutes=5), '30m': timedelta(minutes=30), 
            '1h': timedelta(hours=1), '2h': timedelta(hours=2), 
            '5h': timedelta(hours=5), '12h': timedelta(hours=12), 
            '24h': timedelta(days=1), '5d': timedelta(days=5), 
            '10d': timedelta(days=10), '15d': timedelta(days=15), 
            '30d': timedelta(days=30), '6months': timedelta(days=182), 
            '1y': timedelta(days=365)
        }

        updated_count = 0
        for idx, row in df_log.iterrows():
            if not pd.isna(row['Actual_Price']):
                continue  # Already evaluated
                
            log_time = pd.to_datetime(row['Log_Timestamp'])
            horizon = row['Horizon']
            target_time = log_time + delta_map.get(horizon, timedelta(hours=1))
            
            if datetime.now() >= target_time:
                actual_price = current_live_price
                error_pct = ((actual_price - row['Predicted_Price']) / row['Predicted_Price']) * 100
                
                df_log.at[idx, 'Actual_Price'] = round(actual_price, 2)
                df_log.at[idx, 'Error_Pct'] = round(error_pct, 2)
                updated_count += 1
                print(f"[+] Evaluated {horizon} logged at {log_time}: Actual=${actual_price:.2f} (Error: {error_pct:+.2f}%)")

        if updated_count > 0:
            df_log.to_csv(self.log_file, index=False)
            with open(self.log_file, 'a') as f:
                f.write('\n')
            print(f"[+] Successfully updated {updated_count} matured prediction(s).")
        else:
            print("[*] No newly matured predictions found.")


if __name__ == "__main__":
    BrentEvaluatorTask().run()