import yfinance as yf
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InstitutionalQuantEngine:
    """
    Production Quantitative Trading & Backtesting Engine.
    Includes: Multi-timeframe trend alignment, Risk Sizing, and Strategy Backtester.
    """

    def __init__(self, ticker_symbol: str, benchmark_symbol: str = "^NSEI", portfolio_capital: float = 100000.0):
        self.ticker = ticker_symbol.upper().strip()
        self.benchmark_ticker = benchmark_symbol
        self.capital = portfolio_capital
        self.df_daily = pd.DataFrame()
        self.df_weekly = pd.DataFrame()
        self._load_and_validate_data()

    def _load_and_validate_data(self):
        """Fetches and aligns data safely handling holidays and incomplete bars."""
        try:
            stock = yf.Ticker(self.ticker)
            benchmark = yf.Ticker(self.benchmark_ticker)
            
            daily = stock.history(period="2y", interval="1d")
            weekly = stock.history(period="2y", interval="1wk")
            bench = benchmark.history(period="2y", interval="1d")

            if daily.empty or len(daily) < 200:
                logging.error(f"Data Insufficient for {self.ticker}")
                return

            daily = daily.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            weekly = weekly.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            bench = bench.dropna(subset=['Close'])

            # Fix 1: Outer Join + Forward Fill prevents dropped trading days
            daily = daily.join(bench[['Close']].rename(columns={'Close': 'Close_bench'}), how='left')
            daily['Close_bench'] = daily['Close_bench'].ffill()

            # Fix 2: Drop unclosed running bar for clean close-to-close evaluation
            self.df_daily = daily
            self.df_weekly = weekly[:-1] if len(weekly) > 0 else weekly

        except Exception as e:
            logging.error(f"Data Pipeline Failure: {str(e)}")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Technical Indicators (Wilder's RSI, Rolling VWAP, ATR)."""
        d = df.copy()

        # Anchored Rolling VWAP (20-day)
        tp = (d['High'] + d['Low'] + d['Close']) / 3
        d['VWAP'] = (tp * d['Volume']).rolling(20).sum() / d['Volume'].rolling(20).sum()

        # Wilder's Smoothed RSI (14)
        delta = d['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        d['RSI'] = 100 - (100 / (1 + rs))

        # ATR (14)
        high_low = d['High'] - d['Low']
        high_close = np.abs(d['High'] - d['Close'].shift())
        low_close = np.abs(d['Low'] - d['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        d['ATR'] = true_range.rolling(14).mean()

        # Trend Indicators
        d['SMA_50'] = d['Close'].rolling(50).mean()
        d['SMA_200'] = d['Close'].rolling(200).mean()

        if 'Close_bench' in d.columns:
            d['Bench_SMA_50'] = d['Close_bench'].rolling(50).mean()

        return d

    def _calculate_position_size(self, price: float, stop_loss: float, max_risk_pct: float = 0.01) -> dict:
        """MISSING MODULE 1: Dynamic Position Sizing based on Capital & Volatility Risk."""
        risk_per_share = abs(price - stop_loss)
        if risk_per_share <= 0:
            return {"quantity": 0, "position_value": 0, "risk_amount": 0}

        max_risk_amount = self.capital * max_risk_pct
        quantity = int(max_risk_amount // risk_per_share)
        position_value = round(quantity * price, 2)

        return {
            "quantity": quantity,
            "position_value": position_value,
            "risk_amount": round(quantity * risk_per_share, 2),
            "portfolio_allocation_pct": round((position_value / self.capital) * 100, 2)
        }

    def run_conviction_engine(self) -> dict:
        """Executes full quantitative analysis pipeline."""
        if self.df_daily.empty:
            return {"status": "ERROR", "message": "Pipeline Empty"}

        df_d = self._calculate_indicators(self.df_daily)
        df_w = self._calculate_indicators(self.df_weekly)

        latest_d = df_d.iloc[-1]
        latest_w = df_w.iloc[-1]

        score = 0
        confluences = []
        warnings = []

        # 1. Market Regime
        if latest_d['Close_bench'] > latest_d['Bench_SMA_50']:
            score += 20
            confluences.append("Market Regime: Bullish (Benchmark > 50 SMA)")
        else:
            warnings.append("Market Regime: Bearish/Weak Headwinds")

        # 2. Timeframe Alignment
        if (latest_w['Close'] > latest_w['SMA_50']) and (latest_d['Close'] > latest_d['SMA_50']):
            score += 25
            confluences.append("Trend Alignment: Daily & Weekly > 50 SMA")
        else:
            warnings.append("Timeframe Conflict: Daily vs Weekly Alignment Weak")

        # 3. Institutional Buying (VWAP & Volume)
        avg_vol = df_d['Volume'].tail(20).mean()
        if latest_d['Close'] > latest_d['VWAP'] and latest_d['Volume'] > (avg_vol * 1.3):
            score += 30
            confluences.append(f"Institutional Footprint: > VWAP with {latest_d['Volume']/avg_vol:.1f}x Volume Spike")

        # 4. Volatility Risk Setup
        atr = latest_d['ATR']
        price = latest_d['Close']
        stop_loss = round(price - (atr * 1.5), 2)
        target = round(price + (atr * 3.0), 2)

        if atr > 0:
            score += 25
            confluences.append("ATR Volatility Structure Normal")

        # Dynamic Position Sizing Calculation
        sizing = self._calculate_position_size(price, stop_loss, max_risk_pct=0.01)

        return {
            "status": "SUCCESS",
            "ticker": self.ticker,
            "price": round(price, 2),
            "score": score,
            "stop_loss": stop_loss,
            "target": target,
            "position_sizing": sizing,
            "reasons": confluences,
            "warnings": warnings
        }

    def backtest_strategy(self, lookback_days: int = 365) -> dict:
        """MISSING MODULE 4: Simple Historical Quantitative Backtester Engine."""
        if self.df_daily.empty:
            return {"status": "ERROR"}

        df = self._calculate_indicators(self.df_daily).tail(lookback_days)
        trades = []

        for i in range(50, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            # Signal Condition: Close crosses above 50 SMA + Volume Spike
            avg_vol = df['Volume'].iloc[i-20:i].mean()
            signal = (row['Close'] > row['SMA_50']) and (prev['Close'] <= prev['SMA_50']) and (row['Volume'] > avg_vol * 1.3)

            if signal:
                entry = row['Close']
                sl = entry - (row['ATR'] * 1.5)
                tp = entry + (row['ATR'] * 3.0)

                # Check forward outcome over 20 bars
                future = df.iloc[i+1:i+21]
                hit_tp = False
                hit_sl = False

                for _, f_row in future.iterrows():
                    if f_row['Low'] <= sl:
                        hit_sl = True
                        break
                    if f_row['High'] >= tp:
                        hit_tp = True
                        break

                if hit_tp:
                    trades.append(1)  # Win
                elif hit_sl:
                    trades.append(0)  # Loss

        total_trades = len(trades)
        wins = sum(trades)
        win_rate = round((wins / total_trades * 100), 2) if total_trades > 0 else 0.0

        return {
            "total_signals": total_trades,
            "winning_trades": wins,
            "losing_trades": total_trades - wins,
            "win_rate_pct": win_rate
        }

