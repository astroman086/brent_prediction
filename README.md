# 🛢️ Brent Crude Oil Price Predictor & Evaluator

An automated machine learning pipeline that predicts Brent Crude oil proxy prices (`BZ=F`) across multiple short-term and long-term horizons, logs the predictions, and automatically evaluates/grades them against live market data using **GitHub Actions**.

---

## 🚀 Key Features

* **Dual-Window Architecture:** 
  * **Short-Term (Intraday):** Uses 60 days of hourly data (`BZ=F`) for micro-momentum, volatility, RSI, and moving averages (`5m`, `30m`, `1h`, `2h`, `5h`, `12h`, `24h`).
  * **Long-Term (Macro):** Uses maximum daily historical data to capture swing trends and macro patterns (`5d`, `10d`, `15d`, `30d`, `6months`, `1y`).
* **Live Energy Sentiment Analysis:** Scans energy news RSS feeds (Google News OPEC searches, OilPrice.com) and evaluates geopolitical sentiment using NLTK VADER.
* **Random Forest Regressor:** Employs tuned machine learning models (`scikit-learn`) for forecasting price targets and percentage changes.
* **Fully Automated (GitHub Actions):** Runs continuously in the cloud for free—predicting every 10 minutes and grading matured predictions hourly, committing updates directly back to the repository CSV log.

---

## 📁 Repository Structure

```text
your-repo-name/
├── brent_predictor.py        # Fetches data, runs ML models, and logs predictions
├── brent_evaluator.py        # Compares pending predictions against live prices and grades them
├── brent_predictions_log.csv # Central tracking log (auto-updated by GitHub Actions)
└── .github/
    └── workflows/
        ├── predict.yml       # Cron schedule for runner (Every 10 minutes)
        └── evaluate.yml      # Cron schedule for evaluator (Every hour)
