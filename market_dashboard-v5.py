#!/usr/bin/env python3
"""
Indian Stock Market Dashboard & Derivative Risk Modeling Suite
--------------------------------------------------------------
A production-grade Streamlit + Plotly interactive application.
To run: streamlit run market_dashboard.py
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm

# Try to import mibian for institutional-grade option pricing
try:
    import mibian
    MIBIAN_AVAILABLE = True
except ImportError:
    MIBIAN_AVAILABLE = False

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Nifty & Commodity Derivative Risk Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Caching wrappers to optimize load speed and reduce network latency
@st.cache_data
def get_cached_historical_feed(ticker):
    acq = MarketDataAcquisition()
    return acq.get_historical_feed(ticker)


# Custom styling for professional finance appearance
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1e1e1e;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 1. QUANT ENGINE & SIMULATION CLASSES
# =====================================================================

class TechnicalStrategyEngine:
    @staticmethod
    def calculate_sma(series, window):
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_bollinger_bands(series, window=20, num_std=2):
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return middle, upper, lower

    @classmethod
    def apply_signals(cls, df, ma_short=20, ma_long=50):
        df = df.copy()
        df['SMA_Short'] = cls.calculate_sma(df['Close'], ma_short)
        df['SMA_Long'] = cls.calculate_sma(df['Close'], ma_long)
        df['RSI'] = cls.calculate_rsi(df['Close'], 14)
        df['BB_Mid'], df['BB_Upper'], df['BB_Lower'] = cls.calculate_bollinger_bands(df['Close'], 20)

        signals = []
        for idx in range(len(df)):
            if idx < max(ma_long, 20):
                signals.append("HOLD")
                continue
                
            close = df['Close'].iloc[idx]
            sma_s = df['SMA_Short'].iloc[idx]
            sma_l = df['SMA_Long'].iloc[idx]
            rsi = df['RSI'].iloc[idx]
            bb_upper = df['BB_Upper'].iloc[idx]
            bb_lower = df['BB_Lower'].iloc[idx]
            
            buy_votes = 0
            sell_votes = 0
            
            if sma_s > sma_l:
                buy_votes += 1
            elif sma_s < sma_l:
                sell_votes += 1
                
            if rsi < 30:
                buy_votes += 1.5
            elif rsi > 70:
                sell_votes += 1.5
                
            if close < bb_lower:
                buy_votes += 1
            elif close > bb_upper:
                sell_votes += 1
                
            if buy_votes > sell_votes and buy_votes >= 1.5:
                signals.append("BUY")
            elif sell_votes > buy_votes and sell_votes >= 1.5:
                signals.append("SELL")
            else:
                signals.append("HOLD")
                
        df['Signal'] = signals
        return df


class BlackScholesEngine:
    @staticmethod
    def calculate_greeks(S, K, r, t, sigma, option_type='c'):
        if S <= 0 or K <= 0 or t <= 0 or sigma <= 0:
            return {'price': 0.0, 'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

        if MIBIAN_AVAILABLE:
            try:
                # mibian expects interest rates and volatilities as percentages
                interest_pct = r * 100.0
                volatility_pct = sigma * 100.0
                days_to_expiration = t * 365.0
                
                c = mibian.BS([S, K, interest_pct, days_to_expiration], volatility=volatility_pct)
                
                if option_type.lower() == 'c':
                    return {
                        'price': float(c.callPrice),
                        'delta': float(c.callDelta),
                        'gamma': float(c.gamma),
                        'theta': float(c.callTheta),  # mibian's theta is already daily
                        'vega': float(c.vega / 100.0)  # mibian's vega is for 1% change, matching our format
                    }
                else:
                    return {
                        'price': float(c.putPrice),
                        'delta': float(c.putDelta),
                        'gamma': float(c.gamma),
                        'theta': float(c.putTheta),   # mibian's theta is already daily
                        'vega': float(c.vega / 100.0)  # mibian's vega is for 1% change, matching our format
                    }
            except Exception:
                pass

        # Fallback to high-accuracy custom NumPy calculation if mibian is not installed
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        if option_type.lower() == 'c':
            price = S * cdf_d1 - K * np.exp(-r * t) * cdf_d2
            delta = cdf_d1
            theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * cdf_d2
        else:
            price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = cdf_d1 - 1.0
            theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2)

        gamma = pdf_d1 / (S * sigma * np.sqrt(t))
        vega = S * np.sqrt(t) * pdf_d1

        return {
            'price': float(price),
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta / 365.0),  # converted to daily
            'vega': float(vega / 100.0)    # per 1% vol change
        }


class MarketRiskCalculator:
    @staticmethod
    def calculate_log_returns(prices):
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def categorize_risk_bracket(annual_vol):
        if annual_vol < 0.15:
            return "LOW RISK (Conservative)"
        elif annual_vol < 0.35:
            return "MEDIUM RISK (Moderate)"
        else:
            return "HIGH RISK (Speculative)"

    @classmethod
    def calculate_value_at_risk(cls, prices, confidence_level=0.95, days=1, investment=10000):
        returns = cls.calculate_log_returns(prices)
        if len(returns) < 5:
            return 0.0, 0.0
            
        mean_ret = returns.mean()
        std_ret = returns.std()

        # Parametric VaR
        z_score = norm.ppf(confidence_level)
        parametric_var_pct = -(mean_ret * days - z_score * std_ret * np.sqrt(days))
        parametric_var_amt = max(0.0, parametric_var_pct * investment)

        # Historical VaR
        sorted_returns = np.sort(returns)
        cutoff_idx = int((1.0 - confidence_level) * len(sorted_returns))
        historical_var_pct = -sorted_returns[max(0, cutoff_idx)]
        historical_var_amt = max(0.0, historical_var_pct * investment)

        return float(parametric_var_amt), float(historical_var_amt)

    @classmethod
    def run_monte_carlo_paths(cls, last_price, annual_vol, annual_return=0.08, days=30, simulations=1000):
        dt = 1.0 / 252.0
        mu = annual_return
        sigma = annual_vol

        prices = np.zeros((days + 1, simulations))
        prices[0] = last_price

        for t in range(1, days + 1):
            Z = np.random.normal(0, 1, simulations)
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
        return prices


class MarketDataAcquisition:
    def __init__(self):
        self.tickers_catalog = {
            'Indices': {
                '^NSEI': 'Nifty 50 Index (NSE)',
                '^BSESN': 'BSE Sensex Index (BSE)'
            },
            'ETFs (Commodity & Equity)': {
                'GOLDBEES.NS': 'Nippon India Gold ETF (Gold Proxy)',
                'SILVERBEES.NS': 'Nippon India Silver ETF (Silver Proxy)',
                'NIFTYBEES.NS': 'Nippon India Nifty 50 ETF'
            },
            'Stocks/Sectors': {
                'RELIANCE.NS': 'Reliance Industries (Energy / Conglomerate)',
                'TCS.NS': 'Tata Consultancy Services (IT)',
                'HDFCBANK.NS': 'HDFC Bank (Financials)',
                'INFY.NS': 'Infosys Limited (IT)',
                'ICICIBANK.NS': 'ICICI Bank (Financials)'
            }
        }

    def generate_simulated_ohlcv(self, ticker, start_days=252, base_price=150.0, annual_vol=0.22):
        np.random.seed(hash(ticker) % (2**32))
        date_today = datetime.now()
        dates = [date_today - timedelta(days=x) for x in range(start_days)]
        dates.reverse()

        prices = []
        current = base_price
        dt = 1.0 / 252.0
        drift = 0.08

        for _ in range(start_days):
            change = np.exp((drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * np.random.normal())
            current *= change
            prices.append(current)

        df = pd.DataFrame(index=dates)
        df.index.name = 'Date'
        df['Close'] = prices
        df['Open'] = df['Close'] * (1.0 + np.random.normal(0, 0.005, start_days))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1.0 + np.abs(np.random.normal(0, 0.008, start_days)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1.0 - np.abs(np.random.normal(0, 0.008, start_days)))
        df['Volume'] = (np.random.poisson(1000000, start_days)).astype(float)
        
        return df

    def get_historical_feed(self, ticker):
        # Graceful synthetic generator mirroring Indian market pricing structures
        base_price = 100.0
        volatility = 0.20
        
        if 'NSEI' in ticker:
            base_price, volatility = 24350.0, 0.11
        elif 'BSESN' in ticker:
            base_price, volatility = 79800.0, 0.11
        elif 'GOLDBEES.NS' in ticker:
            base_price, volatility = 68.5, 0.13
        elif 'SILVERBEES.NS' in ticker:
            base_price, volatility = 95.0, 0.17
        elif 'NIFTYBEES.NS' in ticker:
            base_price, volatility = 265.0, 0.12
        elif 'RELIANCE.NS' in ticker:
            base_price, volatility = 3120.0, 0.18
        elif 'TCS.NS' in ticker:
            base_price, volatility = 4250.0, 0.16
        elif 'HDFCBANK.NS' in ticker:
            base_price, volatility = 1680.0, 0.17
        elif 'INFY.NS' in ticker:
            base_price, volatility = 1840.0, 0.20
        elif 'ICICIBANK.NS' in ticker:
            base_price, volatility = 1210.0, 0.18

        return self.generate_simulated_ohlcv(ticker, base_price=base_price, annual_vol=volatility)

    def get_options_chain_feed(self, ticker, spot_price, r_free=0.068):
        strikes = []
        # Find step size based on price magnitude
        if spot_price > 10000:
            step = 100.0
        elif spot_price > 1000:
            step = 50.0
        elif spot_price > 100:
            step = 10.0
        else:
            step = 2.5
            
        rounded_spot = round(spot_price / step) * step
        for i in range(-5, 6):
            strikes.append(rounded_spot + (i * step))

        chain_data = []
        days_to_expiry = 30
        t_years = days_to_expiry / 365.0
        sigma = 0.18  # Implied Volatility anchor

        for strike in strikes:
            # Volatility smile curve
            smile_iv = sigma + 0.0005 * ((strike - spot_price) / step) ** 2
            
            call_greeks = BlackScholesEngine.calculate_greeks(spot_price, strike, r_free, t_years, smile_iv, 'c')
            put_greeks = BlackScholesEngine.calculate_greeks(spot_price, strike, r_free, t_years, smile_iv, 'p')
            
            call_oi = int(120000 / (1.0 + 0.25 * abs(strike - spot_price) / step))
            put_oi = int(100000 / (1.0 + 0.25 * abs(strike - spot_price) / step))

            chain_data.append({
                'Strike': strike,
                'Call_Price': round(call_greeks['price'], 2),
                'Call_Delta': round(call_greeks['delta'], 3),
                'Call_Gamma': round(call_greeks['gamma'], 4),
                'Call_Theta': round(call_greeks['theta'], 3),
                'Call_Open_Interest': call_oi,
                'Put_Price': round(put_greeks['price'], 2),
                'Put_Delta': round(put_greeks['delta'], 3),
                'Put_Gamma': round(put_greeks['gamma'], 4),
                'Put_Theta': round(put_greeks['theta'], 3),
                'Put_Open_Interest': put_oi,
                'Implied_Volatility_Pct': round(smile_iv * 100, 2)
            })

        return pd.DataFrame(chain_data)


# =====================================================================
# 2. APP FLOW EXECUTIVE & RENDERERS
# =====================================================================

def main():
    st.title("🇮🇳 Indian Market Derivative Tracker & Risk Model")
    st.markdown("Automated risk grader, option chain solver, technical indicators, and predictive Monte Carlo simulators.")

    # Instantiate classes
    data_acq = MarketDataAcquisition()

    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.header("⚙️ Configuration Panel")
    
    # 1. Ticker selection organized by category
    categories = list(data_acq.tickers_catalog.keys())
    selected_cat = st.sidebar.selectbox("Market Category", categories, index=2)
    
    ticker_dict = data_acq.tickers_catalog[selected_cat]
    ticker_display = {f"{ticker} ({name})": ticker for ticker, name in ticker_dict.items()}
    selected_disp = st.sidebar.selectbox("Select Asset Ticker", list(ticker_display.keys()), index=0)
    ticker = ticker_display[selected_disp]
    ticker_name = ticker_dict[ticker]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Strategy Parameters")
    ma_short = st.sidebar.slider("SMA Short Window", min_value=5, max_value=30, value=20, step=1)
    ma_long = st.sidebar.slider("SMA Long Window", min_value=30, max_value=100, value=50, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Monte Carlo & Risk Settings")
    sim_days = st.sidebar.slider("Simulate Forward Days", min_value=5, max_value=60, value=30, step=5)
    sim_count = st.sidebar.slider("Number of Random Walks", min_value=500, max_value=5000, value=1500, step=500)
    investment_amt = st.sidebar.number_input("Investment Baseline (INR ₹)", min_value=1000, max_value=10000000, value=10000, step=5000)
    r_free_rate = st.sidebar.slider("Risk-Free Interest Rate (%)", min_value=4.0, max_value=10.0, value=6.8, step=0.1) / 100.0

    # Fetch daily OHLCV data
    with st.spinner(f"Loading daily data for {ticker}..."):
        df = get_cached_historical_feed(ticker)
        df_sig = TechnicalStrategyEngine.apply_signals(df, ma_short=ma_short, ma_long=ma_long)
        
    last_row = df_sig.iloc[-1]
    last_price = last_row['Close']
    prev_price = df_sig['Close'].iloc[-2]
    daily_change = ((last_price - prev_price) / prev_price) * 100.0
    
    # Run risk & volatility math
    log_returns = MarketRiskCalculator.calculate_log_returns(df_sig['Close'])
    daily_vol = log_returns.std()
    annualized_vol = daily_vol * math.sqrt(252)
    risk_bracket = MarketRiskCalculator.categorize_risk_bracket(annualized_vol)

    # Parametric & Historical Value at Risk
    p_var, h_var = MarketRiskCalculator.calculate_value_at_risk(
        df_sig['Close'], confidence_level=0.95, days=1, investment=investment_amt
    )

    # Run Monte Carlo
    paths = MarketRiskCalculator.run_monte_carlo_paths(
        last_price=last_price, annual_vol=annualized_vol, annual_return=r_free_rate, days=sim_days, simulations=sim_count
    )
    terminal_prices = paths[-1]
    prob_up = float(np.sum(terminal_prices > last_price) / sim_count * 100.0)
    median_term_price = np.median(terminal_prices)
    worst_case_price = np.percentile(terminal_prices, 5)
    worst_case_loss_pct = ((last_price - worst_case_price) / last_price) * 100.0

    # --- TOP ROW PERFORMANCE KPI CARD DISPLAY ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Spot Close Price (INR)</div>
            <div class="metric-value">₹{last_price:,.2f}</div>
            <div style="color: {'green' if daily_change >= 0 else 'red'}; font-size:14px; font-weight:bold;">
                {daily_change:+.2f}% (Daily Change)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #28a745;">
            <div class="metric-label">Technical Trend Signal</div>
            <div class="metric-value" style="color: {'#28a745' if last_row['Signal']=='BUY' else ('#dc3545' if last_row['Signal']=='SELL' else '#ffc107')};">
                {last_row['Signal']}
            </div>
            <div style="color: #6c757d; font-size:14px;">RSI (14): {last_row['RSI']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ffc107;">
            <div class="metric-label">Risk Tier Grading</div>
            <div class="metric-value" style="font-size:18px;">{risk_bracket}</div>
            <div style="color: #6c757d; font-size:14px;">Annual Volatility: {annualized_vol*100.0:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #17a2b8;">
            <div class="metric-label">Probability of upside</div>
            <div class="metric-value">{prob_up:.1f}%</div>
            <div style="color: #6c757d; font-size:14px;">Median: ₹{median_term_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- MAIN INTERFACE TABS ---
    t_overview, t_derivatives, t_probability, t_heatmap = st.tabs([
        "📊 Trend & Technical Indicators", 
        "⛓️ Derivative Options & Greeks", 
        "🎲 Portfolio Risk & Predictive Models",
        "🗺️ Sector Technical & Risk Heatmap"
    ])

    # -------------------------------------------------------------
    # TAB 1: HISTORICAL PLOT & INDICATORS
    # -------------------------------------------------------------
    with t_overview:
        st.subheader(f"Historical Trend Analysis: {ticker_name}")
        
        # Plotly Subplots: Price & Volume + RSI
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.08, 
                            row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df_sig.index,
            open=df_sig['Open'],
            high=df_sig['High'],
            low=df_sig['Low'],
            close=df_sig['Close'],
            name="OHLC Price"
        ), row=1, col=1)

        # SMAs
        fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig['SMA_Short'], name=f'SMA {ma_short}', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig['SMA_Long'], name=f'SMA {ma_long}', line=dict(color='blue', width=1.5)), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig['BB_Upper'], name='Bollinger Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig['BB_Lower'], name='Bollinger Lower', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)

        # RSI Subplot
        fig.add_trace(go.Scatter(x=df_sig.index, y=df_sig['RSI'], name='RSI', line=dict(color='purple', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_sig.index, y=[70]*len(df_sig), name='Overbought (70)', line=dict(color='red', width=1, dash='dot'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_sig.index, y=[30]*len(df_sig), name='Oversold (30)', line=dict(color='green', width=1, dash='dot'), showlegend=False), row=2, col=1)

        fig.update_layout(
            title_text=f"{ticker} Daily Technical Indicator Board",
            xaxis_rangeslider_visible=False,
            height=600,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI Value", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show Historical Technical Data Table"):
            st.dataframe(df_sig[['Open', 'High', 'Low', 'Close', 'SMA_Short', 'SMA_Long', 'RSI', 'Signal']].tail(50))


    # -------------------------------------------------------------
    # TAB 2: OPTIONS CHAIN & GREEKS
    # -------------------------------------------------------------
    with t_derivatives:
        st.subheader("Options Derivatives Sentinel (30 Days to Expiration)")
        
        # Load options chain synthetic data
        options_chain = data_acq.get_options_chain_feed(ticker, last_price, r_free=r_free_rate)
        
        # Sentiment metrics
        total_call_oi = options_chain['Call_Open_Interest'].sum()
        total_put_oi = options_chain['Put_Open_Interest'].sum()
        pcr_oi = total_put_oi / float(total_call_oi) if total_call_oi > 0 else 0.0
        sentiment = "BULLISH" if pcr_oi < 0.7 else ("BEARISH" if pcr_oi > 1.0 else "NEUTRAL")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Calls Open Interest", f"{total_call_oi:,} contracts")
        col2.metric("Total Puts Open Interest", f"{total_put_oi:,} contracts")
        col3.metric("Derivative Put-Call Ratio (PCR)", f"{pcr_oi:.2f}")
        col4.metric("Implied Sentiment Rating", sentiment)

        st.markdown("---")
        
        # Open interest comparisons by strike
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            x=options_chain['Strike'],
            y=options_chain['Call_Open_Interest'],
            name='Call Open Interest',
            marker_color='red'
        ))
        fig_oi.add_trace(go.Bar(
            x=options_chain['Strike'],
            y=options_chain['Put_Open_Interest'],
            name='Put Open Interest',
            marker_color='green'
        ))
        fig_oi.update_layout(
            title="Open Interest (OI) Distribution across Strike Prices",
            xaxis_title="Option Strike Price (₹)",
            yaxis_title="Open Interest (Number of Contracts)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_oi, use_container_width=True)

        # Greeks Plots
        st.write("### European Option Greeks Valuation (Black-Scholes Model)")
        fig_greeks = make_subplots(rows=1, cols=2, subplot_titles=("Option Delta Sensitivity", "Option Time Decay (Theta Daily)"))
        
        fig_greeks.add_trace(go.Scatter(x=options_chain['Strike'], y=options_chain['Call_Delta'], name='Call Delta', line=dict(color='red', width=2)), row=1, col=1)
        fig_greeks.add_trace(go.Scatter(x=options_chain['Strike'], y=options_chain['Put_Delta'], name='Put Delta', line=dict(color='green', width=2)), row=1, col=1)
        
        fig_greeks.add_trace(go.Scatter(x=options_chain['Strike'], y=options_chain['Call_Theta'], name='Call Theta', line=dict(color='orange', width=2)), row=1, col=2)
        fig_greeks.add_trace(go.Scatter(x=options_chain['Strike'], y=options_chain['Put_Theta'], name='Put Theta', line=dict(color='blue', width=2)), row=1, col=2)
        
        fig_greeks.update_layout(height=400, showlegend=True)
        fig_greeks.update_xaxes(title_text="Strike Price (₹)", row=1, col=1)
        fig_greeks.update_xaxes(title_text="Strike Price (₹)", row=1, col=2)
        fig_greeks.update_yaxes(title_text="Delta Value", row=1, col=1)
        fig_greeks.update_yaxes(title_text="Daily Decay in INR", row=1, col=2)
        st.plotly_chart(fig_greeks, use_container_width=True)

        st.write("#### Raw Options Chain Sheet")
        st.dataframe(options_chain)


    # -------------------------------------------------------------
    # TAB 3: RISK ANALYSIS & PROBABILITY MODELS
    # -------------------------------------------------------------
    with t_probability:
        st.subheader("Statistical Portfolio Risk & Probability Models")
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc3545;">
                <div class="metric-label">1-Day Parametric Value at Risk (VaR)</div>
                <div class="metric-value">₹{p_var:,.2f}</div>
                <div style="color: #6c757d; font-size:13px;">
                    At 95% Confidence Interval for ₹{investment_amt:,} portfolio allocation.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc3545;">
                <div class="metric-label">1-Day Historical Value at Risk (VaR)</div>
                <div class="metric-value">₹{h_var:,.2f}</div>
                <div style="color: #6c757d; font-size:13px;">
                    Evaluated from historical return quantiles directly.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc3545;">
                <div class="metric-label">95% Worst-Case Downside Limit</div>
                <div class="metric-value" style="color:#dc3545;">-{worst_case_loss_pct:.1f}%</div>
                <div style="color: #6c757d; font-size:13px;">
                    Terminal Price floor: ₹{worst_case_price:,.2f} over {sim_days} days.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Monte Carlo Paths Visualizer
        st.write(f"### {sim_days}-Day Monte Carlo Path Simulations ({sim_count} Walks)")
        
        # Plot up to 80 paths to keep chart responsive
        paths_to_plot = min(80, sim_count)
        fig_paths = go.Figure()
        
        time_axis = list(range(sim_days + 1))
        
        for p in range(paths_to_plot):
            fig_paths.add_trace(go.Scatter(
                x=time_axis, 
                y=paths[:, p], 
                mode='lines', 
                line=dict(width=0.8), 
                opacity=0.3,
                showlegend=False
            ))
            
        # Add Median and 5th percentile paths
        median_path = np.median(paths, axis=1)
        worst_path = np.percentile(paths, 5, axis=1)
        
        fig_paths.add_trace(go.Scatter(x=time_axis, y=median_path, name='Median Price Trend', line=dict(color='blue', width=3)))
        fig_paths.add_trace(go.Scatter(x=time_axis, y=worst_path, name='95% Downside Floor', line=dict(color='red', width=3, dash='dash')))
        
        fig_paths.update_layout(
            xaxis_title="Simulation Time Horizon (Days)",
            yaxis_title="Asset Price (₹)",
            height=450
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        # Terminal Price distribution histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=terminal_prices,
            nbinsx=40,
            marker_color='#17a2b8',
            opacity=0.75,
            name="Terminal Price"
        ))
        # Add a line representing initial spot price
        fig_hist.add_shape(
            type="line", line=dict(color="orange", width=2, dash="dash"),
            x0=last_price, x1=last_price, y0=0, y1=1, yref="paper"
        )
        # Add shape representing worst-case 5% floor
        fig_hist.add_shape(
            type="line", line=dict(color="red", width=2),
            x0=worst_case_price, x1=worst_case_price, y0=0, y1=1, yref="paper"
        )
        fig_hist.update_layout(
            title="Terminal Simulated Price Probability Distribution Grid",
            xaxis_title="Final Price (₹)",
            yaxis_title="Walk Count Density",
            height=350
        )
        st.plotly_chart(fig_hist, use_container_width=True)


    # -------------------------------------------------------------
    # TAB 4: SECTOR TECHNICAL & RISK HEATMAP
    # -------------------------------------------------------------
    with t_heatmap:
        st.subheader("🗺️ Sector Technical & Risk Heatmap Grid")
        st.markdown(
            "This multi-dimensional heat map displays current technical momentum, "
            "volatility, downside risk, and Monte Carlo probability metrics for all 10 assets "
            "across three Indian sectors. **High heat (Green)** represents bullishness/low risk/buying triggers, "
            "while **low heat (Red)** flags overbought levels, high volatility, and downside risk."
        )

        with st.spinner("Calculating multi-sector metrics..."):
            heatmap_data = []
            for cat, tickers_dict in data_acq.tickers_catalog.items():
                for tk, tk_name in tickers_dict.items():
                    df_t = get_cached_historical_feed(tk)
                    df_sig_t = TechnicalStrategyEngine.apply_signals(df_t, ma_short=ma_short, ma_long=ma_long)
                    last_row_t = df_sig_t.iloc[-1]
                    last_price_t = last_row_t['Close']
                    
                    # Technical metrics
                    dev_sma_s = ((last_price_t - last_row_t['SMA_Short']) / last_row_t['SMA_Short']) * 100.0
                    dev_sma_l = ((last_price_t - last_row_t['SMA_Long']) / last_row_t['SMA_Long']) * 100.0
                    rsi_t = last_row_t['RSI']
                    
                    # Volatility & VaR
                    log_returns_t = MarketRiskCalculator.calculate_log_returns(df_sig_t['Close'])
                    vol_t = log_returns_t.std() * math.sqrt(252)
                    p_var_t, _ = MarketRiskCalculator.calculate_value_at_risk(
                        df_sig_t['Close'], confidence_level=0.95, days=1, investment=10000
                    )
                    p_var_pct_t = (p_var_t / 10000.0) * 100.0
                    
                    # Monte Carlo probability
                    paths_t = MarketRiskCalculator.run_monte_carlo_paths(
                        last_price=last_price_t, annual_vol=vol_t, annual_return=r_free_rate, days=sim_days, simulations=min(500, sim_count)
                    )
                    prob_up_t = float(np.sum(paths_t[-1] > last_price_t) / paths_t.shape[1] * 100.0)
                    
                    heatmap_data.append({
                        'Ticker': tk,
                        'Name': tk_name.split(' (')[0],
                        'Category': cat,
                        'RSI (14)': rsi_t,
                        'SMA Short Dev %': dev_sma_s,
                        'SMA Long Dev %': dev_sma_l,
                        'Volatility %': vol_t * 100.0,
                        'Daily VaR %': p_var_pct_t,
                        'Upside Prob %': prob_up_t
                    })
            
            df_hm = pd.DataFrame(heatmap_data)
            
            # Scale metrics from 0 to 1 for heatmap color map
            df_scaled = df_hm.copy()
            cols_to_scale = ['RSI (14)', 'SMA Short Dev %', 'SMA Long Dev %', 'Volatility %', 'Daily VaR %', 'Upside Prob %']
            
            for col in cols_to_scale:
                col_min = df_hm[col].min()
                col_max = df_hm[col].max()
                if col_max != col_min:
                    if col == 'RSI (14)':
                        df_scaled[col] = (col_max - df_hm[col]) / (col_max - col_min)
                    elif col in ['Volatility %', 'Daily VaR %']:
                        df_scaled[col] = (col_max - df_hm[col]) / (col_max - col_min)
                    else:
                        df_scaled[col] = (df_hm[col] - col_min) / (col_max - col_min)
                else:
                    df_scaled[col] = 0.5

            y_labels = [f"{row['Ticker']} ({row['Name']})" for _, row in df_hm.iterrows()]
            x_labels = ['RSI (14)', 'Short SMA Dev %', 'Long SMA Dev %', 'Annual Volatility %', '1-Day VaR %', '10d Upside Prob %']
            
            z_matrix = []
            text_matrix = []
            
            for idx, row in df_scaled.iterrows():
                z_row = [
                    row['RSI (14)'],
                    row['SMA Short Dev %'],
                    row['SMA Long Dev %'],
                    row['Volatility %'],
                    row['Daily VaR %'],
                    row['Upside Prob %']
                ]
                z_matrix.append(z_row)
                
                raw_row = df_hm.iloc[idx]
                text_row = [
                    f"RSI: {raw_row['RSI (14)']:.1f}",
                    f"Dev: {raw_row['SMA Short Dev %']:+.1f}%",
                    f"Dev: {raw_row['SMA Long Dev %']:+.1f}%",
                    f"Vol: {raw_row['Volatility %']:.1f}%",
                    f"VaR: {raw_row['Daily VaR %']:.2f}%",
                    f"Prob: {raw_row['Upside Prob %']:.1f}%"
                ]
                text_matrix.append(text_row)

            fig_hm = go.Figure(data=go.Heatmap(
                z=z_matrix,
                x=x_labels,
                y=y_labels,
                colorscale='RdYlGn',
                colorbar=dict(title='Signal Intensity', ticks='', showticklabels=False),
                text=text_matrix,
                texttemplate="%{text}",
                hoverinfo="text"
            ))
            
            fig_hm.update_layout(
                title="Multi-Sector Technical Trend and Risk Gradient Matrix",
                height=550,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(side="top"),
                margin=dict(t=100, b=50, l=150, r=50)
            )
            st.plotly_chart(fig_hm, use_container_width=True)



if __name__ == '__main__':
    main()
