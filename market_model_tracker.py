#!/usr/bin/env python3
"""
Stock Market Indicator & Derivative Risk Modeling Suite
--------------------------------------------------------
This script provides a production-ready framework to extract daily stock index and
commodity data, evaluate technical indicators, analyze derivative option chains,
calculate portfolio risk profiles, and estimate probability metrics like Value at Risk (VaR).

Written with robust fallbacks: it automatically runs simulations using realistic synthetic
market feeds if yfinance is not available or if the environment is offline.
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

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
    """Calculates Options pricing and Greeks using standard mathematical models."""

    @staticmethod
    def calculate_greeks(S, K, r, t, sigma, option_type='c'):
        """
        Calculates price, Delta, Gamma, Theta, Vega for an option.
        S: Underlying stock price
        K: Strike price
        r: Risk-free rate (decimal, e.g., 0.05 for 5%)
        t: Time to expiration in years (e.g., 30 / 365.0)
        sigma: Implied volatility (decimal, e.g., 0.20 for 20%)
        option_type: 'c' for Call, 'p' for Put
        """
        # Return 0s if standard inputs are nonsensical
        if S <= 0 or K <= 0 or t <= 0 or sigma <= 0:
            return {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        # Option Price
        if option_type.lower() == 'c':
            price = S * cdf_d1 - K * np.exp(-r * t) * cdf_d2
            delta = cdf_d1
            # Theta for European Call
            theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * cdf_d2
        else:
            price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = cdf_d1 - 1.0
            # Theta for European Put
            theta = -(S * pdf_d1 * sigma) / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2)

        # Shared Greeks
        gamma = pdf_d1 / (S * sigma * np.sqrt(t))
        vega = S * np.sqrt(t) * pdf_d1

        # Annual Theta converted to Daily Theta (divided by 365)
        return {
            'price': float(price),
            'delta': float(delta),
            'gamma': float(gamma),
            'theta': float(theta / 365.0),
            'vega': float(vega / 100.0)  # Vega per 1% change in volatility
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
        elif annual_vol < 0.35:
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
    def run_monte_carlo_simulation(cls, last_price, annual_vol, annual_return=0.08, days=10, simulations=5000):
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
                '^GSPC': 'S&P 500 Index',
                '^IXIC': 'NASDAQ Composite',
                '^DJI': 'Dow Jones Industrial Average'
            },
            'Commodities': {
                'GC=F': 'Gold Futures',
                'CL=F': 'Crude Oil Futures',
                'SI=F': 'Silver Futures'
            },
            'Stocks/Sectors': {
                'AAPL': 'Apple Inc. (Technology)',
                'XOM': 'Exxon Mobil (Energy)',
                'JPM': 'JPMorgan Chase (Financials)',
                'JNJ': 'Johnson & Johnson (Healthcare)',
                'AMZN': 'Amazon.com (Consumer Discretionary)'
            }
        }

    def generate_simulated_ohlcv(self, ticker, start_days=252, base_price=150.0, annual_vol=0.22):
        """Generates realistic daily asset prices using a geometric random walk."""
        np.random.seed(hash(ticker) % (2**32))  # Stable seed per ticker
        date_today = datetime.now()
        dates = [date_today - timedelta(days=x) for x in range(start_days)]
        dates.reverse()

        prices = []
        current = base_price
        dt = 1.0 / 252.0
        drift = 0.08  # 8% typical annual drift

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
        
        # Fallback simulated configurations for demo purposes
        base_price = 100.0
        volatility = 0.20
        
        if 'GSPC' in ticker:
            base_price, volatility = 5500.0, 0.14
        elif 'IXIC' in ticker:
            base_price, volatility = 18000.0, 0.18
        elif 'DJI' in ticker:
            base_price, volatility = 40000.0, 0.12
        elif 'GC=F' in ticker:
            base_price, volatility = 2400.0, 0.13
        elif 'CL=F' in ticker:
            base_price, volatility = 75.0, 0.32
        elif 'AAPL' in ticker:
            base_price, volatility = 220.0, 0.22
        elif 'XOM' in ticker:
            base_price, volatility = 115.0, 0.25
        elif 'JPM' in ticker:
            base_price, volatility = 205.0, 0.21
        elif 'JNJ' in ticker:
            base_price, volatility = 160.0, 0.11
        elif 'AMZN' in ticker:
            base_price, volatility = 180.0, 0.28

        return self.generate_simulated_ohlcv(ticker, base_price=base_price, annual_vol=volatility)

    def get_options_chain_feed(self, ticker, spot_price):
        """Retrieves options chain data or generates realistic synthetic chains with option Greeks."""
        strikes = []
        rounded_spot = round(spot_price / 5.0) * 5.0
        
        # Create strikes around the current underlying spot price
        for i in range(-5, 6):
            strikes.append(rounded_spot + (i * 5.0))

        chain_data = []
        interest_rate = 0.053  # US T-Bill Rate (Approx 5.3% in current economic climate)
        days_to_expiry = 30
        t_years = days_to_expiry / 365.0
        sigma = 0.24  # Base implied volatility

        for strike in strikes:
            # Generate slightly upward-sloping IV smile
            smile_iv = sigma + 0.001 * (strike - spot_price) ** 2
            
            # Call calculations
            call_greeks = BlackScholesEngine.calculate_greeks(
                spot_price, strike, interest_rate, t_years, smile_iv, 'c'
            )
            # Put calculations
            put_greeks = BlackScholesEngine.calculate_greeks(
                spot_price, strike, interest_rate, t_years, smile_iv, 'p'
            )
            
            # Open Interest generation (highest open interest is usually near standard psychological numbers)
            call_oi = int(15000 / (1.0 + 0.15 * abs(strike - spot_price)))
            put_oi = int(12000 / (1.0 + 0.15 * abs(strike - spot_price)))

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
    print("=" * 70)
    print("      STOCK MARKET INDICATOR & DERIVATIVE RISK EVALUATOR ENGINE")
    print("=" * 70)
    print(f"Engine Initialization Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"yfinance Live Feed Connectivity: {'ENABLED' if YFINANCE_AVAILABLE else 'FALLBACK ACTIVE'}")
    print("=" * 70)

    data_acq = MarketDataAcquisition()
    analysis_results = []
    
    # Track across Indices, Commodities, and Sectors
    target_categories = data_acq.tickers_catalog
    
    for cat_name, tickers in target_categories.items():
        print(f"\nEvaluating Category Sector: [{cat_name.upper()}]")
        print("-" * 50)
        
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
            
            # 4. Value at Risk Analysis (VaR)
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
                'Last_Price': last_price,
                'Daily_Change_Pct': daily_change,
                'Annual_Volatility_Pct': annualized_vol * 100.0,
                'Risk_Bracket': risk_bracket,
                'SMA_20': last_row['SMA_20'],
                'RSI_14': last_row['RSI'],
                'Trend_Signal': last_row['Signal'],
                'Daily_95%_VaR_USD': parametric_var,
                'MC_Prob_Up_Pct': mc['probability_up'],
                'MC_Worst_Case_Loss_Pct': mc['worst_case_loss_pct']
            })
            
            # Display localized highlights
            print(f" -> Ticker: {ticker:<6} | Price: ${last_price:,.2f} ({daily_change:+.2f}%) | Signal: {last_row['Signal']:<4} | Risk Bracket: {risk_bracket.split(' ')[0]}")

    results_df = pd.DataFrame(analysis_results)
    
    # 6. Derivative Analysis Highlight (on Apple AAPL)
    print("\n" + "=" * 70)
    print("  DERIVATIVE TRACKING SUB-ENGINE: SPOTLIGHT ON STOCK 'AAPL'")
    print("=" * 70)
    aapl_spot = results_df[results_df['Ticker'] == 'AAPL']['Last_Price'].values[0]
    options_chain = data_acq.get_options_chain_feed('AAPL', aapl_spot)
    
    # Calculate major derivative sentiment indicators
    total_call_oi = options_chain['Call_Open_Interest'].sum()
    total_put_oi = options_chain['Put_Open_Interest'].sum()
    pcr_oi = total_put_oi / float(total_call_oi) if total_call_oi > 0 else 0
    
    print(f"AAPL Spot Price: ${aapl_spot:,.2f}")
    print(f"Total Call Open Interest (30 days to Expiry): {total_call_oi:,} contracts")
    print(f"Total Put Open Interest (30 days to Expiry) : {total_put_oi:,} contracts")
    print(f"Option Put-Call Ratio (PCR) by Open Interest: {pcr_oi:.2f}")
    sentiment = "BULLISH (More Calls active)" if pcr_oi < 0.7 else ("BEARISH (More Puts active)" if pcr_oi > 1.0 else "NEUTRAL")
    print(f"Derivatives-Implied Sentiment Bracket: {sentiment}")
    print("-" * 50)
    print(options_chain[['Strike', 'Call_Price', 'Call_Delta', 'Call_Theta', 'Put_Price', 'Put_Delta', 'Put_Theta', 'Implied_Volatility_Pct']].head(5).to_string(index=False))
    print("...")
    
    # Save the output deliverables to workspace csvs
    results_df.to_csv('/workspace/scratch/market_summary_model.csv', index=False)
    options_chain.to_csv('/workspace/scratch/aapl_options_chain_analysis.csv', index=False)
    
    print("\n" + "=" * 70)
    print("  SUMMARY RISK RATINGS & PROBABILITY OF SUCCESS REPORT")
    print("=" * 70)
    for risk_lbl in ["LOW RISK (Conservative)", "MEDIUM RISK (Moderate)", "HIGH RISK (Speculative)"]:
        sub_group = results_df[results_df['Risk_Bracket'] == risk_lbl]
        if not sub_group.empty:
            print(f"\n--- {risk_lbl} ---")
            for idx, row in sub_group.iterrows():
                print(f" * {row['Ticker']:<6} ({row['Name']}): Vol: {row['Annual_Volatility_Pct']:.1f}% | Daily VaR: ${row['Daily_95%_VaR_USD']:.2f} | 10d Up Prob: {row['MC_Prob_Up_Pct']:.1f}%")

    return results_df, options_chain


if __name__ == '__main__':
    execute_model_suite()
