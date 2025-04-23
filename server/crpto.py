import time
import pandas as pd
import numpy as np
from binance.enums import *
from binance.client import Client
import ta
import csv
import logging
import os
import hashlib
import boto3
from datetime import datetime, timedelta
import random
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


BINANCE_API_KEY = "LZD68mFDteis5Ra4ZZCXbMq41f95aP0YdGgWZg1Kkbw9vHtC2idARFL3a2K6eaUf"
BINANCE_API_SECRET = "pFqRF0qeZxO2Uewe39EcFOySAd8cnCcfU3H9Qyqm84DnjXs9TJt5kxZgU3hqPW90"

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)  

price = client.get_symbol_ticker(symbol="BTCUSDT")
print(f"Testnet BTC price: {price['price']}")

# Enhanced logging configuration

logging.basicConfig(
    filename='scalping_bot_sim.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    
)
logging.info("Starting the trading bot script...")
logging.debug("Debug message test - If you see this, logging works!")



# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

# S3 client initialization with error handling
try:
    s3 = boto3.client('s3')
except Exception as e:
    logging.error(f"Failed to initialize S3 client: {e}")
    s3 = None

@dataclass
class TradingConfig:
    """Trading configuration with type hints and validation"""
    SYMBOL: str = "BTCUSDT"
    BASE_CAPITAL: float = 500.0
    LEVERAGE: int = 10
    QUANTITY: float = 0.0005
    PROFIT_TARGET: float = 0.01
    STOP_LOSS: float = 0.005
    DAILY_LOSS_LIMIT: float = 0.0
    VOLATILITY_THRESHOLD: float = 0.05
    MIN_LIQUIDITY: float = 200_000_000
    TRAILING_STOP: float = 0.0025
    KLINES_INTERVAL: int = 300
    FEE_RATE: float = 0.000375
    SIM_SPEED: int = 86400

    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.BASE_CAPITAL <= 0:
            raise ValueError("BASE_CAPITAL must be positive")
        if self.LEVERAGE <= 0:
            raise ValueError("LEVERAGE must be positive")
        if self.QUANTITY <= 0:
            raise ValueError("QUANTITY must be positive")
        if self.PROFIT_TARGET <= 0:
            raise ValueError("PROFIT_TARGET must be positive")
        if self.STOP_LOSS <= 0:
            raise ValueError("STOP_LOSS must be positive")
    
CONFIG = TradingConfig() 
class TradingBot:
    def __init__(self, api_key: str, api_secret: str, simulate: bool = False):
        """
        Initialize the TradingBot.

        Args:
        api_key (str): Binance API key.
        api_secret (str): Binance API secret.
        simulate (bool, optional): Whether to simulate trading. Defaults to False.
        """
        
        self.simulate = simulate
        self.config = CONFIG
        self.client = Client(api_key, api_secret, testnet=True)
        self.api = None
        self.bm = None
        
        # Initialize trading state
        self._initial_capital = CONFIG.BASE_CAPITAL
        self._balance = CONFIG.BASE_CAPITAL
        self._profit_buffer = 0.0
        self._daily_loss = 0.0
        self._in_position = False
        self._last_price = 0.0
        self._strategy = "none"
        self._trade_log: List[List] = []
        
        # Market state
        self._current_price = 60000.0
        self._bid_ask_ratio = 1.0
        self._total_depth = 0.0
        self._depth_gap = 0.0
        
        # Technical analysis state
        self._klines_df = pd.DataFrame()
        self._last_kline_time = 0
        self._last_order_time = 0
        self._sim_time = 0
        
        # Validate configuration
        CONFIG.validate()
        
        if not simulate:
            self._initialize_live_trading()
        else:
            self._simulate_market()

    def _initialize_live_trading(self) -> None:
     """Initialize live trading with Binance Testnet API connections"""
    try:
        required_env_vars = [
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET"
        ]

        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

        from binance.client import Client
        from binance import BinanceSocketManager

        logging.info("Successfully connected to Binance Testnet API")
        
    except Exception as e:
        logging.critical(f"Failed to initialize live trading: {e}")
        raise

    def process_depth(self, msg: Dict) -> None:
        """Process market depth data with enhanced error handling"""
        try:
            if not isinstance(msg, dict) or 'bids' not in msg or 'asks' not in msg:
                raise ValueError("Invalid depth data structure")

            bids = sum(float(b[1]) for b in msg['bids'])
            asks = sum(float(a[1]) for a in msg['asks'])

            if bids < 0 or asks < 0:
                raise ValueError("Negative depth detected")

            self.bid_ask_ratio = bids / asks if asks > 0 else 1.0
            self.total_depth = bids + asks
            
            bid_price = float(msg['bids'][0][0])
            ask_price = float(msg['asks'][0][0])
            self.depth_gap = (ask_price - bid_price) / bid_price

        except Exception as e:
            logging.error(f"Depth processing error: {e}")
            self.bid_ask_ratio = 1.0
            self.total_depth = 0.0
            self.depth_gap = 0.0

    def simulate_market(self) -> None:
        """Improved market simulation with realistic price movements"""
        try:
            # Apply more realistic price movement using random walk
            price_change = np.random.normal(0, self.current_price * 0.001)  # 0.1% standard deviation
            self.current_price = max(0, self.current_price + price_change)
            
            # Simulate market depth with more realistic values
            self.bid_ask_ratio = np.random.lognormal(0, 0.1)  # Log-normal distribution for ratio
            self.total_depth = np.random.uniform(200_000_000, 500_000_000)
            self.depth_gap = abs(np.random.normal(0.002, 0.0005))  # Normal distribution for gap
            
            # Simulate occasional market events
            if self.sim_time % 5 == 0:
                event_magnitude = random.choice([1.05, 0.95])
                self.current_price *= event_magnitude
                logging.info(f"Market event: Price moved to ${self.current_price:.2f}")
            
            self.sim_time += 1
            self.run_trade_logic()
            
            if self.sim_time < 20:  # 20 simulation days
                time.sleep(1 / CONFIG.SIM_SPEED * 86400)
                self.simulate_market()
                
        except Exception as e:
            logging.error(f"Simulation error: {e}")
            raise

    
    def get_klines(self) -> None:
        """Fetch Binance Testnet candlestick (kline) data"""
        try:
            if self.simulate:
                self._generate_simulated_klines()
            else:
                klines = self.client.get_klines(
                    symbol=self.config.SYMBOL,
                    interval=Client.KLINE_INTERVAL_5MINUTE,
                    limit=100
                )
                self.klines_df = pd.DataFrame(klines, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "num_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
                self.klines_df["close"] = self.klines_df["close"].astype(float)
                self.klines_df["open"] = self.klines_df["open"].astype(float)
                self.klines_df["high"] = self.klines_df["high"].astype(float)
                self.klines_df["low"] = self.klines_df["low"].astype(float)
                logging.info("Retrieved Binance Testnet market data")
        except Exception as e:
            logging.error(f"Failed to fetch klines from Binance Testnet: {e}")

    def _generate_simulated_klines(self) -> None:
        """Generate simulated kline data"""
        times = np.arange(self.sim_time * 60, (self.sim_time + 1) * 60, 60)
        closes = [self.current_price * (1 + np.random.normal(0, 0.005)) for _ in times]
        self.klines_df = pd.DataFrame({
            'timestamp': times,
            'open': closes,
            'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in closes],
            'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in closes],
            'close': closes,
            'volume': np.random.uniform(100, 1000, len(times))
        })
        self._calculate_technical_indicators()

    def _calculate_technical_indicators(self) -> None:
        """Calculate technical indicators"""
        try:
            self.klines_df['close'] = self.klines_df['close'].astype(float)
            self.klines_df['EMA9'] = ta.trend.ema_indicator(self.klines_df['close'], window=9)
            self.klines_df['EMA21'] = ta.trend.ema_indicator(self.klines_df['close'], window=21)
            self.klines_df['RSI'] = ta.momentum.rsi(self.klines_df['close'], window=14)
            self.klines_df['ATR'] = ta.volatility.average_true_range(
                self.klines_df['high'],
                self.klines_df['low'],
                self.klines_df['close'],
                window=14
            )
            self.klines_df['VWAP'] = (
                self.klines_df['close'] * self.klines_df['volume']
            ).cumsum() / self.klines_df['volume'].cumsum()
        except Exception as e:
            logging.error(f"Failed to calculate technical indicators: {e}")

    def update_risk(self) -> None:
        """Update risk parameters based on current balance"""
        try:
            if self.balance <= self.config.BASE_CAPITAL:
                self.config.DAILY_LOSS_LIMIT = 0.0
            else:
                self.profit_buffer = self.balance - self.config.BASE_CAPITAL
                self.config.DAILY_LOSS_LIMIT = min(0.05, self.profit_buffer / self.balance)
        except Exception as e:
            logging.error(f"Failed to update risk parameters: {e}")
            self.config.DAILY_LOSS_LIMIT = 0.0

    def log_trade(self) -> None:
        """Log trade data"""
        if not self.trade_log:
            return
        try:
            with open("trade_log_sim.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.trade_log)
            self.trade_log.clear()
        except Exception as e:
            logging.error(f"Failed to log trades: {e}")

    def run(self) -> None:
        """Main run method"""
        try:
            if not self.simulate:
                logging.info(f"Starting live trading bot with ${self.config.BASE_CAPITAL}...")
                from twisted.internet import reactor
                reactor.run()
            else:
                logging.info(f"Starting simulation with ${self.config.BASE_CAPITAL}...")
                self.simulate_market()
                logging.info(f"Simulation ended. Final balance: ${self.balance:.2f}")
        except Exception as e:
            logging.critical(f"Fatal error in trading bot: {e}")
            raise


if __name__ == "__main__":
    bot = TradingBot(BINANCE_API_KEY, BINANCE_API_SECRET)
    print("Bot instance:", bot)
    bot.get_klines()
