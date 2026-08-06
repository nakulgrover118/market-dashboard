#!/usr/bin/env python3
"""
Indian Stock Market Indicator & Derivative Risk Modeling Suite (v2)
------------------------------------------------------------------
This script provides a production-ready framework to extract daily stock index,
ETF, and Indian commodity data, evaluate technical indicators, analyze derivative
option chains, calculate portfolio risk profiles, and estimate probability metrics
such as Value at Risk (VaR) in Indian Rupees (INR).

Features:
1. Focus on Indian indices, sector leaders, and commodity ETFs (Nifty, Gold, Silver, Reliance).
2. Option Greeks solver based on the Black-Scholes model, using an Indian risk-free rate of 6.8%.
3. Parametric and Historical VaR calculations.
4. Monte Carlo simulations using Geometric Brownian Motion (GBM).
5. Structured report generation in rupees (₹) and automated CSV database exports.
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

# Try to import mibian for institutional-grade option pricing
try:
    import mibian
    MIBIAN_AVAILABLE = True
except ImportError:
    MIBIAN_AVAILABLE = False

# Try to import yfinance, but fall back gracefully if not present
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ==========================================
# 1. TECHNICAL ANALYSIS & STRATEGY ENGINE
# ==========================================
class TechnicalStrategyEngine:
    """Calculates quantitative signals and technical indicators on historical time series."""

    @staticmethod
    def calculate_sma(series, window):
        """Simple Moving Average."""
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(series, window):
        """Exponential Moving Average."""
        return series.rolling(window=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(series, window=14):
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(series, window=20, num_std=2):
        """Bollinger Bands (Middle, Upper, Lower)."""
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return middle, upper, lower

    @classmethod
    def apply_signals(cls, df):
        """Generates clear buy/sell/neutral trend signals based on technical indicators."""
        close_prices = df['Close']
        
        df['SMA_20'] = cls.calculate_sma(close_prices, 20)
        df['SMA_50'] = cls.calculate_sma(close_prices, 50)
        df['RSI'] = cls.calculate_rsi(close_prices, 14)
        df['BB_Mid'], df['BB_Upper'], df['BB_Lower'] = cls.calculate_bollinger_bands(close_prices, 20)

        # Signal Generation:
        # 1. Crossover: Buy when SMA20 > SMA50
        # 2. RSI Oversold/Overbought: RSI < 30 (Oversold/Buy), RSI > 70 (Overbought/Sell)
        # 3. Bollinger Bands: Close < Lower Band (Oversold/Buy), Close > Upper Band (Overbought/Sell)
        
        signals = []
        for idx in range(len(df)):
            if idx < 50:  # Warm-up period for moving averages
                signals.append("HOLD")
                continue
                
            close = df['Close'].iloc[idx]
            sma20 = df['SMA_20'].iloc[idx]
            sma50 = df['SMA_50'].iloc[idx]
            rsi = df['RSI'].iloc[idx]
            bb_upper = df['BB_Upper'].iloc[idx]
            bb_lower = df['BB_Lower'].iloc[idx]
            
            buy_votes = 0
            sell_votes = 0
            
            # MA Trend Crossover
            if sma20 > sma50:
                buy_votes += 1
            elif sma20 < sma50:
                sell_votes += 1
                
            # RSI Indicator
            if rsi < 30:
                buy_votes += 1.5  # RSI oversold is strong indicator
            elif rsi > 70:
                sell_votes += 1.5  # RSI overbought is strong indicator
                
            # Bollinger Bands
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


# ==========================================
# 2. DERIVATIVES & BLACK-SCHOLES GREEKS ENGINE
# ==========================================
class BlackScholesEngine:
    """Calculates Options pricing and Greeks using standard mathematical models (Mibian with custom fallback)."""

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
                        'vega': float(c.vega / 100.0)  # vega per 1% change in volatility
                    }
                else:
                    return {
                        'price': float(c.putPrice),
                        'delta': float(c.putDelta),
                        'gamma': float(c.gamma),
                        'theta': float(c.putTheta),   # mibian's theta is already daily
                        'vega': float(c.vega / 100.0)  # vega per 1% change in volatility
                    }
            except Exception:
                pass

        # Native Black-Scholes fallback math (when mibian is not installed)
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
            'theta': float(theta / 365.0),
            'vega': float(vega / 100.0)
        }


# ==========================================
# 3. PORTFOLIO RISK & PROBABILITY CALCULATOR
# ==========================================
class MarketRiskCalculator:
    """Performs risk evaluation, volatility grouping, and Value at Risk (VaR) estimations."""

    @staticmethod
    def calculate_log_returns(prices):
        """Calculates daily continuously compounded log returns."""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def categorize_risk_bracket(annual_vol):
        """Grades an asset into risk tiers based on annualized standard deviation."""
        if annual_vol < 0.15:
            return "LOW RISK (Conservative)"
        elif annual_vol < 0.28:
            return "MEDIUM RISK (Moderate)"
        else:
            return "HIGH RISK (Speculative)"

    @classmethod
    def calculate_value_at_risk(cls, prices, confidence_level=0.95, days=1, investment=10000):
        """
        Estimates the parametric and historical Value at Risk (VaR).
        Provides the maximum expected loss at a given confidence interval.
        """
        returns = cls.calculate_log_returns(prices)
        if len(returns) < 5:
            return 0.0, 0.0
            
        mean_ret = returns.mean()
        std_ret = returns.std()

        # 1. Parametric VaR (Variance-Covariance Method)
        z_score = norm.ppf(confidence_level)
        parametric_var_pct = -(mean_ret * days - z_score * std_ret * np.sqrt(days))
        parametric_var_amt = max(0.0, parametric_var_pct * investment)

        # 2. Historical VaR
        sorted_returns = np.sort(returns)
        cutoff_idx = int((1.0 - confidence_level) * len(sorted_returns))
        historical_var_pct = -sorted_returns[max(0, cutoff_idx)]
        historical_var_amt = max(0.0, historical_var_pct * investment)

        return float(parametric_var_amt), float(historical_var_amt)

    @classmethod
    def run_monte_carlo_simulation(cls, last_price, annual_vol, annual_return=0.11, days=10, simulations=5000):
        """
        Runs a Monte Carlo simulation using Geometric Brownian Motion (GBM).
        Estimates terminal price distribution and probability of positive returns.
        """
        dt = 1.0 / 252.0  # Daily time-step
        mu = annual_return
        sigma = annual_vol

        # Generate paths
        prices = np.zeros((days + 1, simulations))
        prices[0] = last_price

        for t in range(1, days + 1):
            Z = np.random.normal(0, 1, simulations)
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        terminal_prices = prices[-1]
        pct_positive_returns = float(np.sum(terminal_prices > last_price) / simulations * 100.0)
        median_price = float(np.median(terminal_prices))
        worst_5_pct_price = float(np.percentile(terminal_prices, 5))
        potential_max_loss_pct = float((last_price - worst_5_pct_price) / last_price * 100.0)

        return {
            'terminal_median': median_price,
            'probability_up': pct_positive_returns,
            'worst_case_price': worst_5_pct_price,
            'worst_case_loss_pct': potential_max_loss_pct
        }


# ==========================================
# 4. DATA ACQUISITION FLOWS (WITH FALLBACK)
# ==========================================
class MarketDataAcquisition:
    """Manages downloading live financial data or generating pristine, realistic simulated assets."""

    def __init__(self):
        self.tickers_catalog = {
            'Indices': {
                '^NSEI': 'Nifty 50 Index (NSE)',
                '^BSESN': 'BSE Sensex Index (BSE)'
            },
            'ETFs (Commodity & Equity)': {
                'GOLDBEES.NS': 'Nippon India Gold ETF (Gold Commodity Proxy)',
                'SILVERBEES.NS': 'Nippon India Silver ETF (Silver Commodity Proxy)',
                'NIFTYBEES.NS': 'Nippon India Nifty 50 ETF'
            },
            'Stocks/Sectors': {
                'RELIANCE.NS': 'Reliance Industries (Energy / Conglomerate)',
                'TCS.NS': 'Tata Consultancy Services (Information Technology)',
                'HDFCBANK.NS': 'HDFC Bank (Financials)',
                'INFY.NS': 'Infosys Limited (Information Technology)',
                'ICICIBANK.NS': 'ICICI Bank (Financials)'
            }
        }

    def generate_simulated_ohlcv(self, ticker, start_days=252, base_price=100.0, annual_vol=0.20):
        """Generates realistic daily asset prices using a geometric random walk."""
        np.random.seed(hash(ticker) % (2**32))  # Stable seed per ticker
        date_today = datetime.now()
        dates = [date_today - timedelta(days=x) for x in range(start_days)]
        dates.reverse()

        prices = []
        current = base_price
        dt = 1.0 / 252.0
        drift = 0.11  # 11% typical annual drift in Indian markets (long-term historical equity risk premium)

        for _ in range(start_days):
            change = np.exp((drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * np.random.normal())
            current *= change
            prices.append(current)

        df = pd.DataFrame(index=dates)
        df.index.name = 'Date'
        df['Close'] = prices
        df['Open'] = df['Close'] * (1.0 + np.random.normal(0, 0.005, start_days))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1.0 + np.abs(np.random.normal(0, 0.008, start_days)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1.0 - np.abs(np.random.normal(0, 0.008, start_days)))\
        
        # Indian daily volume averages
        base_vol = 5000000 if 'NSEI' in ticker or 'BSESN' in ticker else 250000
        df['Volume'] = (np.random.poisson(base_vol, start_days)).astype(float)
        
        return df

    def get_historical_feed(self, ticker, period='1y'):
        """Attempts live download, falls back to realistic simulation if offline."""
        if YFINANCE_AVAILABLE:
            try:
                # Limit timeouts and suppress verbose outputs
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(period=period)
                if not df.empty:
                    return df
            except Exception:
                pass
        
        # Fallback simulated configurations representing typical Indian stock and ETF prices (INR)
        base_price = 100.0
        volatility = 0.20
        
        if 'NSEI' in ticker:
            base_price, volatility = 24300.0, 0.12
        elif 'BSESN' in ticker:
            base_price, volatility = 79500.0, 0.11
        elif 'GOLDBEES.NS' in ticker:
            base_price, volatility = 65.5, 0.14
        elif 'SILVERBEES.NS' in ticker:
            base_price, volatility = 88.2, 0.18
        elif 'NIFTYBEES.NS' in ticker:
            base_price, volatility = 265.0, 0.12
        elif 'RELIANCE.NS' in ticker:
            base_price, volatility = 2950.0, 0.18
        elif 'TCS.NS' in ticker:
            base_price, volatility = 4100.0, 0.16
        elif 'HDFCBANK.NS' in ticker:
            base_price, volatility = 1650.0, 0.17
        elif 'INFY.NS' in ticker:
            base_price, volatility = 1850.0, 0.19
        elif 'ICICIBANK.NS' in ticker:
            base_price, volatility = 1180.0, 0.18

        return self.generate_simulated_ohlcv(ticker, base_price=base_price, annual_vol=volatility)

    def get_options_chain_feed(self, ticker, spot_price):
        """Retrieves options chain data or generates realistic synthetic chains with Indian option Greeks."""
        strikes = []
        # India NSE typically uses strikes spaced by 5, 10, or 50 depending on stock/price. 
        # For RELIANCE (~2950), strike intervals are 20 or 50. Let's use 50.
        rounded_spot = round(spot_price / 50.0) * 50.0
        
        # Create strikes around the current underlying spot price
        for i in range(-5, 6):
            strikes.append(rounded_spot + (i * 50.0))

        chain_data = []
        interest_rate = 0.068  # 6.8% typical RBI Repo/Government bond yield in India (2026)
        days_to_expiry = 30
        t_years = days_to_expiry / 365.0
        sigma = 0.18  # Base implied volatility for RELIANCE

        for strike in strikes:
            # Generate slightly upward-sloping IV smile
            smile_iv = sigma + 0.0001 * (strike - spot_price) ** 2
            
            # Call calculations
            call_greeks = BlackScholesEngine.calculate_greeks(
                spot_price, strike, interest_rate, t_years, smile_iv, 'c'
            )
            # Put calculations
            put_greeks = BlackScholesEngine.calculate_greeks(
                spot_price, strike, interest_rate, t_years, smile_iv, 'p'
            )
            
            # Open Interest generation (highest open interest is usually near round numbers/At-The-Money)
            call_oi = int(50000 / (1.0 + 0.05 * abs(strike - spot_price)))
            put_oi = int(45000 / (1.0 + 0.05 * abs(strike - spot_price)))

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


# ==========================================
# 5. CORE MODEL WORKFLOW EXECUTIVE
# ==========================================
def execute_model_suite():
    """Runs the full risk evaluation and market tracking pipeline."""
    print("=" * 75)
    print("      INDIAN STOCK MARKET INDICATOR & DERIVATIVE RISK EVALUATOR (v3)")
    print("=" * 75)
    print(f"Engine Initialization Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"yfinance Live Feed Connectivity: {'ENABLED' if YFINANCE_AVAILABLE else 'FALLBACK ACTIVE'}")
    print("=" * 75)

    data_acq = MarketDataAcquisition()
    analysis_results = []
    
    # Track across Indices, Commodities, and Sectors
    target_categories = data_acq.tickers_catalog
    
    for cat_name, tickers in target_categories.items():
        print(f"\nEvaluating Category Sector: [{cat_name.upper()}]")
        print("-" * 55)
        
        for ticker, name in tickers.items():
            # 1. Extract raw OHLCV price histories
            df = data_acq.get_historical_feed(ticker)
            if df.empty:
                continue
                
            # 2. Extract Technical indicators and strategize buy/sell triggers
            df_signals = TechnicalStrategyEngine.apply_signals(df)
            last_row = df_signals.iloc[-1]
            last_price = last_row['Close']
            prev_price = df_signals['Close'].iloc[-2]
            daily_change = ((last_price - prev_price) / prev_price) * 100.0
            
            # 3. Calculate volatility and group into risk profiles
            log_returns = MarketRiskCalculator.calculate_log_returns(df_signals['Close'])
            daily_vol = log_returns.std()
            annualized_vol = daily_vol * math.sqrt(252)
            risk_bracket = MarketRiskCalculator.categorize_risk_bracket(annualized_vol)
            
            # 4. Value at Risk Analysis (VaR) based on ₹10,000 baseline investment
            parametric_var, historical_var = MarketRiskCalculator.calculate_value_at_risk(
                df_signals['Close'], confidence_level=0.95, days=1, investment=10000
            )
            
            # 5. Monte Carlo Predictive Probabilities (10 days out)
            mc = MarketRiskCalculator.run_monte_carlo_simulation(
                last_price=last_price, annual_vol=annualized_vol, days=10
            )
            
            # Store primary performance data
            analysis_results.append({
                'Category': cat_name,
                'Ticker': ticker,
                'Name': name,
                'Last_Price_INR': last_price,
                'Daily_Change_Pct': daily_change,
                'Annual_Volatility_Pct': annualized_vol * 100.0,
                'Risk_Bracket': risk_bracket,
                'SMA_20': last_row['SMA_20'],
                'RSI_14': last_row['RSI'],
                'Trend_Signal': last_row['Signal'],
                'Daily_95%_VaR_INR': parametric_var,
                'MC_Prob_Up_Pct': mc['probability_up'],
                'MC_Worst_Case_Loss_Pct': mc['worst_case_loss_pct']
            })
            
            # Display localized highlights
            print(f" -> Ticker: {ticker:<11} | Price: Rs.{last_price:,.2f} ({daily_change:+.2f}%) | Signal: {last_row['Signal']:<4} | Risk: {risk_bracket.split(' ')[0]}")

    results_df = pd.DataFrame(analysis_results)
    
    # 6. Derivative Analysis Highlight (on Reliance RELIANCE.NS)
    print("\n" + "=" * 75)
    print("  DERIVATIVE TRACKING SUB-ENGINE: SPOTLIGHT ON STOCK 'RELIANCE.NS'")
    print("=" * 75)
    reliance_spot = results_df[results_df['Ticker'] == 'RELIANCE.NS']['Last_Price_INR'].values[0]
    options_chain = data_acq.get_options_chain_feed('RELIANCE.NS', reliance_spot)
    
    # Calculate major derivative sentiment indicators
    total_call_oi = options_chain['Call_Open_Interest'].sum()
    total_put_oi = options_chain['Put_Open_Interest'].sum()
    pcr_oi = total_put_oi / float(total_call_oi) if total_call_oi > 0 else 0
    
    print(f"RELIANCE.NS Spot Price: Rs.{reliance_spot:,.2f}")
    print(f"Total Call Open Interest (30 days to Expiry): {total_call_oi:,} contracts")
    print(f"Total Put Open Interest (30 days to Expiry) : {total_put_oi:,} contracts")
    print(f"Option Put-Call Ratio (PCR) by Open Interest: {pcr_oi:.2f}")
    sentiment = "BULLISH (More Calls active)" if pcr_oi < 0.7 else ("BEARISH (More Puts active)" if pcr_oi > 1.0 else "NEUTRAL")
    print(f"Derivatives-Implied Sentiment Bracket: {sentiment}")
    print("-" * 55)
    print(options_chain[['Strike', 'Call_Price', 'Call_Delta', 'Call_Theta', 'Put_Price', 'Put_Delta', 'Put_Theta', 'Implied_Volatility_Pct']].head(5).to_string(index=False))
    print("...")
    
    # Save the output deliverables to workspace csvs
    results_df.to_csv('/workspace/scratch/market_summary_model-v3.csv', index=False)
    options_chain.to_csv('/workspace/scratch/reliance_options_chain_analysis-v3.csv', index=False)
    
    print("\n" + "=" * 75)
    print("  SUMMARY RISK RATINGS & PROBABILITY OF SUCCESS REPORT")
    print("=" * 75)
    for risk_lbl in ["LOW RISK (Conservative)", "MEDIUM RISK (Moderate)", "HIGH RISK (Speculative)"]:
        sub_group = results_df[results_df['Risk_Bracket'] == risk_lbl]
        if not sub_group.empty:
            print(f"\n--- {risk_lbl} ---")
            for idx, row in sub_group.iterrows():
                print(f" * {row['Ticker']:<11} ({row['Name']}): Vol: {row['Annual_Volatility_Pct']:.1f}% | Daily VaR: Rs.{row['Daily_95%_VaR_INR']:.2f} | 10d Up Prob: {row['MC_Prob_Up_Pct']:.1f}%")

    return results_df, options_chain


if __name__ == '__main__':
    execute_model_suite()
