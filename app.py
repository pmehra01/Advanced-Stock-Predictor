import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedStockPredictor:
    def __init__(self, ticker, start_date='2010-01-01', end_date=None):
        """
        Enhanced Stock Price Predictor with multiple models and advanced features
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def fetch_data(self):
        """Enhanced data fetching with error handling and info retrieval"""
        try:
            # Fetch stock data
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            # Get stock info
            try:
                self.stock_info = stock.info
            except:
                self.stock_info = {}
            
            if self.data.empty:
                st.error(f"No data found for ticker {self.ticker}")
                return None
                
            # Clean column names
            self.data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            return self.data
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def advanced_feature_engineering(self):
        """Create comprehensive technical indicators and features"""
        if self.data is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")
        
        df = self.data.copy()
        
        # Basic price features
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = df['Price_Change'] / df['Open'] * 100
        
        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) > period:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        df['Stoch_K'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                        (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        
        # Volume indicators
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['Price_Volume'] = df['Close'] * df['Volume']
            
            # On Balance Volume (OBV)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volatility measures
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['ATR'] = self.calculate_atr(df)
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        # Lagged features
        for lag in range(1, 8):
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else 0
        
        # Time-based features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['IsMonthEnd'] = df.index.is_month_end.astype(int)
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        # Trend indicators
        df['Trend_5'] = np.where(df['Close'] > df['SMA_5'], 1, 0)
        df['Trend_20'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
        df['Trend_50'] = np.where(df['Close'] > df['SMA_50'], 1, 0)
        
        self.data = df.dropna()
        return self.data
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def prepare_features(self):
        """Prepare feature matrix for machine learning"""
        # Select relevant features
        feature_columns = []
        
        # Price-based features
        price_features = ['Price_Range', 'Price_Change_Pct', 'BB_Position', 'BB_Width']
        
        # Technical indicators
        technical_features = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                            'Stoch_K', 'Stoch_D', 'Volatility', 'ATR']
        
        # Moving averages
        ma_features = [col for col in self.data.columns if 'SMA_' in col or 'EMA_' in col]
        
        # Volume features
        volume_features = [col for col in self.data.columns if 'Volume' in col and col != 'Volume']
        
        # Lagged features
        lag_features = [col for col in self.data.columns if 'Lag_' in col]
        
        # Time features
        time_features = ['DayOfWeek', 'Month', 'Quarter', 'IsMonthEnd', 'IsQuarterEnd']
        
        # Trend features
        trend_features = [col for col in self.data.columns if 'Trend_' in col]
        
        # Combine all features
        all_features = (price_features + technical_features + ma_features + 
                       volume_features + lag_features + time_features + trend_features)
        
        # Filter existing columns
        feature_columns = [col for col in all_features if col in self.data.columns]
        
        # Remove any remaining NaN columns
        feature_columns = [col for col in feature_columns if not self.data[col].isna().all()]
        
        X = self.data[feature_columns].fillna(method='ffill').fillna(method='bfill')
        y = self.data['Close']
        
        return X, y, feature_columns
    
    def train_multiple_models(self, X_train, y_train):
        """Train multiple models and compare performance"""
        models_config = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
            'Linear Regression': LinearRegression(),
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        trained_models = {}
        scalers = {}
        
        for name, model in models_config.items():
            try:
                # Scale features for certain models
                if name in ['Support Vector Regression', 'Neural Network']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    scalers[name] = scaler
                else:
                    X_train_scaled = X_train
                    scalers[name] = None
                
                # Train model
                model.fit(X_train_scaled, y_train)
                trained_models[name] = model
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                    
            except Exception as e:
                st.warning(f"Failed to train {name}: {str(e)}")
                continue
        
        self.models = trained_models
        self.scalers = scalers
        return trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # Scale test data if needed
                if self.scalers[name] is not None:
                    X_test_scaled = self.scalers[name].transform(X_test)
                else:
                    X_test_scaled = X_test
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                predictions[name] = y_pred
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'RMSE': np.sqrt(mse)
                }
                
            except Exception as e:
                st.warning(f"Failed to evaluate {name}: {str(e)}")
                continue
        
        return results, predictions
    
    def predict_future(self, days=30, model_name='Random Forest'):
        """Predict future stock prices"""
        if model_name not in self.models:
            st.error(f"Model {model_name} not available")
            return None
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Get last known features
        last_features = self.X.iloc[-1:].copy()
        predictions = []
        
        for day in range(days):
            try:
                # Scale features if needed
                if scaler is not None:
                    features_scaled = scaler.transform(last_features)
                else:
                    features_scaled = last_features
                
                # Make prediction
                pred = model.predict(features_scaled)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified)
                # In a real scenario, you'd need more sophisticated feature updating
                last_features = last_features.copy()
                
            except Exception as e:
                st.warning(f"Error in future prediction: {e}")
                break
        
        return predictions

def create_interactive_charts(data, predictions=None, ticker=""):
    """Create interactive Plotly charts"""
    
    # Main price chart with technical indicators
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['Price & Moving Averages', 'Volume', 'RSI', 'MACD'],
        row_heights=[0.6, 0.2, 0.1, 0.1]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', 
                            line=dict(color='#2E86AB', width=2)), row=1, col=1)
    
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', 
                                line=dict(color='orange')), row=1, col=1)
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', 
                                line=dict(color='red')), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper',
                                line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='BB Lower',
                                line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # Volume
    if 'Volume' in data.columns:
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                            marker_color='lightblue'), row=2, col=1)
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', 
                                line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', 
                                line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', 
                                line=dict(color='red')), row=4, col=1)
    
    fig.update_layout(
        title=f'{ticker} Stock Analysis Dashboard',
        height=800,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_feature_importance_chart(predictor, feature_names):
    """Create feature importance visualization"""
    if not predictor.feature_importance:
        return None
    
    # Get Random Forest importance (if available)
    if 'Random Forest' in predictor.feature_importance:
        importance = predictor.feature_importance['Random Forest']
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True).tail(15)
        
        # Create horizontal bar chart
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 15 Most Important Features (Random Forest)',
            template='plotly_white'
        )
        
        fig.update_layout(height=500)
        return fig
    
    return None

def main():
    # Enhanced page configuration
    st.set_page_config(
        page_title="Advanced Stock Predictor",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .feature-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .model-comparison {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🚀 Advanced Stock Price Predictor</h1>', unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("## 🎯 Trading Dashboard")
        st.markdown("---")
        
        # Quick stock selection with categories
        st.markdown("### 📈 Stock Categories")
        
        stock_categories = {
            "🔥 Tech Giants": ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA'],
            "⚡ Growth Stocks": ['TSLA', 'NFLX', 'ZOOM', 'SHOP', 'SQ', 'ROKU'],
            "🏦 Financial": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V'],
            "🏥 Healthcare": ['JNJ', 'PFE', 'UNH', 'ABBV', 'BMY', 'MRK'],
            "🎮 Entertainment": ['DIS', 'NFLX', 'EA', 'ATVI', 'T', 'VZ']
        }
        
        selected_category = st.selectbox("Choose Category", list(stock_categories.keys()))
        selected_quick_stock = st.selectbox("Quick Select", stock_categories[selected_category])
        
        if st.button(f"Load {selected_quick_stock}", type="secondary"):
            st.session_state.ticker = selected_quick_stock
        
        st.markdown("---")
        
        # Manual ticker input
        ticker = st.text_input(
            '🎯 Enter Stock Ticker', 
            value=st.session_state.get('ticker', 'AAPL'),
            placeholder="e.g., AAPL, GOOGL, TSLA"
        ).upper()
        
        # Enhanced date selection
        st.markdown("### 📅 Analysis Period")
        
        # Preset date ranges
        date_presets = {
            "1 Year": 365,
            "2 Years": 730,
            "3 Years": 1095,
            "5 Years": 1825,
            "Custom": None
        }
        
        date_preset = st.selectbox("Select Period", list(date_presets.keys()), index=2)
        
        if date_preset != "Custom":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=date_presets[date_preset])
        else:
            col1, col2 = st.columns(2)
            start_date = col1.date_input('Start', datetime.now() - timedelta(days=1095))
            end_date = col2.date_input('End', datetime.now())
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("### ⚙️ Analysis Options")
        
        enable_future_prediction = st.checkbox("🔮 Future Predictions", value=True)
        prediction_days = st.slider("Prediction Days", 1, 90, 30) if enable_future_prediction else 30
        
        show_technical_analysis = st.checkbox("📊 Technical Analysis", value=True)
        show_model_comparison = st.checkbox("🤖 Model Comparison", value=True)
        show_feature_importance = st.checkbox("🎯 Feature Importance", value=True)
        
        st.markdown("---")
        
        # Run analysis button
        analyze_button = st.button(
            '🚀 Run Advanced Analysis', 
            type="primary",
            use_container_width=True
        )
    
    # Main content
    if not analyze_button:
        # Enhanced welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h2>🎯 Advanced Features</h2>
                <ul style="text-align: left; font-size: 1.1rem;">
                    <li>🤖 Multiple ML Models (RF, GB, SVM, NN)</li>
                    <li>📈 50+ Technical Indicators</li>
                    <li>🔮 Future Price Predictions</li>
                    <li>📊 Interactive Charts & Dashboards</li>
                    <li>💹 Real-time Stock Information</li>
                    <li>⚡ Performance Comparison</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature showcase
        st.markdown("### ✨ What Makes This Special")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <h3>🤖 5 AI Models</h3>
                <p>Random Forest, Gradient Boosting, SVM, Neural Networks & Linear Regression</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <h3>📊 50+ Features</h3>
                <p>Technical indicators, moving averages, volume analysis & more</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <h3>📈 Interactive Charts</h3>
                <p>Plotly-powered dynamic visualizations with technical overlays</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            
            <div class="metric-container">
                <h3>🔮 Future Predictions</h3>
                <p>Predict stock prices up to 90 days in advance</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Main analysis
        with st.spinner('🔄 Initializing advanced analysis...'):
            predictor = EnhancedStockPredictor(
                ticker=ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Fetch data
                status_text.text('📡 Fetching market data...')
                progress_bar.progress(15)
                data = predictor.fetch_data()
                
                if data is None or data.empty:
                    st.error("❌ Failed to fetch data. Please check ticker and try again.")
                    st.stop()
                
                # Step 2: Feature engineering
                status_text.text('🔧 Engineering 50+ features...')
                progress_bar.progress(30)
                enhanced_data = predictor.advanced_feature_engineering()
                
                # Step 3: Prepare features
                status_text.text('📊 Preparing feature matrix...')
                progress_bar.progress(45)
                X, y, feature_names = predictor.prepare_features()
                
                # Step 4: Split data
                test_size = min(0.2, 100/len(X))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=False, random_state=42
                )
                
                predictor.X = X  # Store for future predictions
                
                # Step 5: Train models
                status_text.text('🤖 Training 5 AI models...')
                progress_bar.progress(70)
                trained_models = predictor.train_multiple_models(X_train, y_train)
                
                # Step 6: Evaluate models
                status_text.text('📈 Evaluating model performance...')
                progress_bar.progress(90)
                results, predictions = predictor.evaluate_models(X_test, y_test)
                
                progress_bar.progress(100)
                status_text.text('✅ Analysis complete!')
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.success(f"🎉 Successfully analyzed {ticker} with {len(enhanced_data)} data points!")
                
                # Stock information
                col1, col2, col3, col4, col5 = st.columns(5)
                
                current_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                
                col1.metric("💰 Current Price", f"${current_price:.2f}", f"{price_change:.2f}")
                col2.metric("📊 Price Change %", f"{price_change_pct:.2f}%")
                col3.metric("📈 52W High", f"${data['High'].max():.2f}")
                col4.metric("📉 52W Low", f"${data['Low'].min():.2f}")
                col5.metric("📅 Data Points", f"{len(data):,}")
                
             # Interactive technical chart
                if show_technical_analysis:
                    st.markdown("### 📊 Interactive Technical Analysis")
                    fig = create_interactive_charts(enhanced_data, ticker=ticker)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Model comparison results
                if show_model_comparison and results:
                    st.markdown("### 🤖 AI Model Performance Comparison")
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results).T
                    results_df = results_df.round(4)
                    
                    # Sort by R2 score (best first)
                    results_df = results_df.sort_values('R2', ascending=False)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Model Rankings")
                        for i, (model, metrics) in enumerate(results_df.iterrows()):
                            color = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                            st.markdown(f"""
                            <div class="model-comparison">
                                <h4>{color} {model}</h4>
                                <p>R² Score: {metrics['R2']:.4f}</p>
                                <p>RMSE: ${metrics['RMSE']:.2f}</p>
                                <p>MAE: ${metrics['MAE']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Performance comparison chart
                        fig_comparison = px.bar(
                            results_df.reset_index(), 
                            x='index', 
                            y='R2',
                            title='Model Performance Comparison (R² Score)',
                            labels={'index': 'Model', 'R2': 'R² Score'},
                            color='R2',
                            color_continuous_scale='viridis'
                        )
                        fig_comparison.update_layout(height=400)
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Detailed metrics table
                    st.markdown("#### 📊 Detailed Performance Metrics")
                    st.dataframe(results_df, use_container_width=True)
                
                # Feature importance
                if show_feature_importance and predictor.feature_importance:
                    st.markdown("### 🎯 Feature Importance Analysis")
                    fig_importance = create_feature_importance_chart(predictor, feature_names)
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Top features summary
                    if 'Random Forest' in predictor.feature_importance:
                        importance_data = list(zip(feature_names, predictor.feature_importance['Random Forest']))
                        importance_data.sort(key=lambda x: x[1], reverse=True)
                        
                        st.markdown("#### 🔝 Top 10 Most Important Features")
                        top_features = importance_data[:10]
                        
                        cols = st.columns(2)
                        for i, (feature, importance) in enumerate(top_features):
                            col_idx = i % 2
                            cols[col_idx].markdown(f"**{i+1}. {feature}**: {importance:.4f}")
                
                # Trading recommendations
                st.markdown("### 💡 AI Trading Insights")
                
                # Technical signals
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Technical Signals")
                    
                    current_data = enhanced_data.iloc[-1]
                    signals = []
                    
                    # RSI signal
                    if 'RSI' in current_data:
                        rsi = current_data['RSI']
                        if rsi > 70:
                            signals.append("🔴 RSI Overbought (Sell Signal)")
                        elif rsi < 30:
                            signals.append("🟢 RSI Oversold (Buy Signal)")
                        else:
                            signals.append("🟡 RSI Neutral")
                    
                    # Moving average signals
                    if 'SMA_20' in current_data and 'SMA_50' in current_data:
                        if current_data['Close'] > current_data['SMA_20'] > current_data['SMA_50']:
                            signals.append("🟢 Above Moving Averages (Bullish)")
                        elif current_data['Close'] < current_data['SMA_20'] < current_data['SMA_50']:
                            signals.append("🔴 Below Moving Averages (Bearish)")
                        else:
                            signals.append("🟡 Mixed Moving Average Signals")
                    
                    # Bollinger Bands
                    if 'BB_Position' in current_data:
                        bb_pos = current_data['BB_Position']
                        if bb_pos > 0.8:
                            signals.append("🔴 Near Upper Bollinger Band (Overbought)")
                        elif bb_pos < 0.2:
                            signals.append("🟢 Near Lower Bollinger Band (Oversold)")
                        else:
                            signals.append("🟡 Within Bollinger Bands")
                    
                    for signal in signals:
                        st.markdown(f"• {signal}")
                
                with col2:
                    st.markdown("#### 🎯 Risk Assessment")
                    
                    # Volatility assessment
                    if 'Volatility' in enhanced_data.columns:
                        recent_vol = enhanced_data['Volatility'].tail(20).mean()
                        vol_percentile = (enhanced_data['Volatility'] < recent_vol).mean() * 100
                        
                        if vol_percentile > 80:
                            st.markdown("🔴 **High Volatility** - Higher risk")
                        elif vol_percentile > 60:
                            st.markdown("🟡 **Moderate Volatility** - Medium risk")
                        else:
                            st.markdown("🟢 **Low Volatility** - Lower risk")
                    
                    # Model agreement
                    if len(results) > 1:
                        r2_scores = [results[model]['R2'] for model in results]
                        avg_r2 = np.mean(r2_scores)
                        std_r2 = np.std(r2_scores)
                        
                        if std_r2 < 0.05:
                            st.markdown("🟢 **High Model Agreement** - Reliable predictions")
                        elif std_r2 < 0.1:
                            st.markdown("🟡 **Moderate Model Agreement** - Some uncertainty")
                        else:
                            st.markdown("🔴 **Low Model Agreement** - High uncertainty")
                        
                        st.markdown(f"Average R² Score: {avg_r2:.4f}")
                        st.markdown(f"Model Consensus: {std_r2:.4f}")
                
                # Additional analytics
                with st.expander("📈 Advanced Analytics", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### 📊 Price Statistics")
                        price_stats = enhanced_data['Close'].describe()
                        for stat, value in price_stats.items():
                            st.markdown(f"**{stat.title()}**: ${value:.2f}")
                    
                    with col2:
                        st.markdown("#### 📈 Returns Analysis")
                        returns = enhanced_data['Close'].pct_change().dropna()
                        st.markdown(f"**Daily Return Mean**: {returns.mean():.4f}")
                        st.markdown(f"**Daily Return Std**: {returns.std():.4f}")
                        st.markdown(f"**Sharpe Ratio**: {returns.mean()/returns.std():.4f}")
                        st.markdown(f"**Max Drawdown**: {(returns.cumsum().cummax() - returns.cumsum()).max():.4f}")
                    
                    with col3:
                        st.markdown("#### 🎯 Model Metrics")
                        if results:
                            best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
                            best_metrics = results[best_model_name]
                            st.markdown(f"**Best Model**: {best_model_name}")
                            st.markdown(f"**R² Score**: {best_metrics['R2']:.4f}")
                            st.markdown(f"**RMSE**: ${best_metrics['RMSE']:.2f}")
                            st.markdown(f"**MAE**: ${best_metrics['MAE']:.2f}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.markdown("**Troubleshooting Tips:**")
                st.markdown("• Check if the ticker symbol is valid")
                st.markdown("• Try a different date range")
                st.markdown("• Ensure stable internet connection")
                st.markdown("• Some tickers may have limited historical data")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>🚀 Advanced Stock Predictor</h4>
        <p>Powered by Machine Learning • Built with Streamlit • Real-time Market Data</p>
        <p><small>⚠️ For educational purposes only. Not financial advice.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()