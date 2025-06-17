````markdown
# 🚀 Advanced Stock Predictor

**Author:** [pmehra01](https://github.com/pmehra01)  

---

## 📊 Overview

**Advanced Stock Predictor** is a Streamlit-based web application that uses machine learning models and technical indicators to analyze and predict stock market prices. It integrates real-time financial data from Yahoo Finance and provides advanced visual dashboards, feature importance insights, and AI-powered forecasting.

---

## 🔍 Features

- 📈 Real-time stock data fetching with `yfinance`
- 🤖 Multiple ML models:
  - Random Forest
  - Gradient Boosting
  - Support Vector Regression (SVR)
  - Neural Network (MLP)
  - Linear Regression
- 🧠 50+ technical indicators including:
  - SMA, EMA, RSI, MACD, Bollinger Bands
  - Volume-based features, Volatility, ATR, OBV, etc.
- 🔮 Future stock price predictions (up to 90 days)
- 📊 Interactive charts using Plotly
- 📉 Model performance comparison (R², RMSE, MAE)
- 🎯 Feature importance visualization (Random Forest)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**:  
  `yfinance`, `numpy`, `pandas`, `scikit-learn`, `plotly`, `matplotlib`, `seaborn`

---

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pmehra01/advanced-stock-predictor.git
   cd advanced-stock-predictor
````

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## ⚠️ Disclaimer

This application is for **educational and informational** purposes only and should not be considered financial advice. Always do your own research before making investment decisions.

---

## 🙌 Credits

Developed by [pmehra01](https://github.com/pmehra01)
