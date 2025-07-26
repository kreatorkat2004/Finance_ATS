import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
import sqlite3
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    position_type: str  

@dataclass
class Trade:
    symbol: str
    action: str  
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0

class RiskManager:
    def __init__(self, max_position_size: float = 0.1, max_drawdown: float = 0.15, 
                 stop_loss: float = 0.05, take_profit: float = 0.15):
        self.max_position_size = max_position_size 
        self.max_drawdown = max_drawdown  
        self.stop_loss = stop_loss 
        self.take_profit = take_profit 
        
    def check_position_size(self, portfolio_value: float, position_value: float) -> bool:
        return position_value <= portfolio_value * self.max_position_size
    
    def check_drawdown(self, current_value: float, peak_value: float) -> bool:
        drawdown = (peak_value - current_value) / peak_value
        return drawdown <= self.max_drawdown
    
    def should_stop_loss(self, entry_price: float, current_price: float, position_type: str) -> bool:
        if position_type == 'long':
            return (entry_price - current_price) / entry_price >= self.stop_loss
        else: 
            return (current_price - entry_price) / entry_price >= self.stop_loss
    
    def should_take_profit(self, entry_price: float, current_price: float, position_type: str) -> bool:
        if position_type == 'long':
            return (current_price - entry_price) / entry_price >= self.take_profit
        else:  
            return (entry_price - current_price) / entry_price >= self.take_profit

class DataProvider(ABC):
    @abstractmethod
    def get_historical_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_real_time_price(self, symbol: str) -> float:
        pass

class AlphaVantageProvider(DataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_historical_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Fetch historical data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.error(f"Error fetching data for {symbol}: {data}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['Time Series (Daily)']).T
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend', 'split']
            df = df.sort_index()
            
            # Filter by period
            if period == '1y':
                df = df[df.index >= datetime.now() - timedelta(days=365)]
            elif period == '6m':
                df = df[df.index >= datetime.now() - timedelta(days=180)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_real_time_price(self, symbol: str) -> float:
        """Get real-time price from Alpha Vantage"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                return float(data['Global Quote']['05. price'])
            else:
                logger.error(f"Error fetching real-time price for {symbol}: {data}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {e}")
            return 0.0

class MomentumStrategy:
    def __init__(self, short_window: int = 20, long_window: int = 50, rsi_period: int = 14):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on momentum strategy"""
        df = self.calculate_indicators(df)
        
        df['signal'] = 0
        df['position'] = 0
        
        conditions_buy = [
            df['sma_short'] > df['sma_long'],  
            df['rsi'] < 70,  
            df['macd'] > df['macd_signal'],  
            df['close'] > df['bb_lower'] 
        ]
        
        conditions_sell = [
            df['sma_short'] < df['sma_long'],  
            df['rsi'] > 30, 
            df['macd'] < df['macd_signal'],  
            df['close'] < df['bb_upper']  
        ]
        
        df.loc[np.all(conditions_buy, axis=0), 'signal'] = 1 
        df.loc[np.all(conditions_sell, axis=0), 'signal'] = -1  
        
        df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        return df

class Backtester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def run_backtest(self, df: pd.DataFrame, strategy: MomentumStrategy) -> Dict:
        """Run backtest on historical data"""
        df = strategy.generate_signals(df)
        
        df['market_return'] = df['close'].pct_change()
        df['strategy_return'] = df['position'].shift(1) * df['market_return']
        
        df['cumulative_market_return'] = (1 + df['market_return']).cumprod()
        df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod()
        
        df['portfolio_value'] = self.initial_capital * df['cumulative_strategy_return']
        
        total_return = df['cumulative_strategy_return'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        
        volatility = df['strategy_return'].std() * np.sqrt(252)
        
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        rolling_max = df['cumulative_strategy_return'].expanding().max()
        drawdown = (df['cumulative_strategy_return'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        winning_trades = df[df['strategy_return'] > 0]['strategy_return'].count()
        total_trades = df[df['strategy_return'] != 0]['strategy_return'].count()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': df['portfolio_value'].iloc[-1],
            'data': df
        }
        
        self.results = results
        return results
    
    def print_results(self):
        """Print backtest results"""
        if not self.results:
            print("No backtest results available")
            return
        
        print("=== BACKTEST RESULTS ===")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${self.results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {self.results['total_return']:.2%}")
        print(f"Annualized Return: {self.results['annualized_return']:.2%}")
        print(f"Volatility: {self.results['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.results['max_drawdown']:.2%}")
        print(f"Win Rate: {self.results['win_rate']:.2%}")
        print(f"Total Trades: {self.results['total_trades']}")

class TradingSystem:
    def __init__(self, data_provider: DataProvider, strategy: MomentumStrategy, 
                 risk_manager: RiskManager, initial_capital: float = 100000):
        self.data_provider = data_provider
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.peak_value = initial_capital
        
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for trade logging"""
        self.conn = sqlite3.connect('trading_system.db')
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                action TEXT,
                quantity REAL,
                price REAL,
                timestamp TEXT,
                pnl REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL,
                entry_price REAL,
                entry_time TEXT,
                position_type TEXT
            )
        ''')
        
        self.conn.commit()
    
    def log_trade(self, trade: Trade):
        """Log trade to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, action, quantity, price, timestamp, pnl)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (trade.symbol, trade.action, trade.quantity, trade.price, 
              trade.timestamp.isoformat(), trade.pnl))
        self.conn.commit()
    
    def execute_trade(self, symbol: str, action: str, quantity: float, price: float) -> bool:
        """Execute a trade"""
        trade_value = quantity * price
        
        if action == 'buy':
            if trade_value > self.current_capital:
                logger.warning(f"Insufficient capital for {symbol} trade")
                return False
            
            if not self.risk_manager.check_position_size(self.current_capital, trade_value):
                logger.warning(f"Position size too large for {symbol}")
                return False
            
            if symbol in self.positions:
                current_pos = self.positions[symbol]
                new_quantity = current_pos.quantity + quantity
                new_entry_price = ((current_pos.quantity * current_pos.entry_price) + 
                                 (quantity * price)) / new_quantity
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    entry_price=new_entry_price,
                    entry_time=current_pos.entry_time,
                    position_type='long'
                )
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    position_type='long'
                )
            
            self.current_capital -= trade_value
            
        elif action == 'sell':
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                logger.warning(f"Insufficient shares to sell for {symbol}")
                return False
            
            position = self.positions[symbol]
            pnl = (price - position.entry_price) * quantity
            
            if position.quantity == quantity:
                del self.positions[symbol]
            else:
                self.positions[symbol].quantity -= quantity
            
            self.current_capital += trade_value
            
            trade = Trade(symbol, action, quantity, price, datetime.now(), pnl)
            self.trades.append(trade)
            self.log_trade(trade)
            
            logger.info(f"Executed {action} {quantity} shares of {symbol} at ${price:.2f}, PnL: ${pnl:.2f}")
            return True
        
        trade = Trade(symbol, action, quantity, price, datetime.now())
        self.trades.append(trade)
        self.log_trade(trade)
        
        logger.info(f"Executed {action} {quantity} shares of {symbol} at ${price:.2f}")
        return True
    
    def check_risk_management(self):
        """Check risk management rules and close positions if necessary"""
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        if not self.risk_manager.check_drawdown(portfolio_value, self.peak_value):
            logger.warning("Maximum drawdown exceeded, closing all positions")
            self.close_all_positions()
            return
        
        positions_to_close = []
        for symbol, position in self.positions.items():
            current_price = self.data_provider.get_real_time_price(symbol)
            
            if (self.risk_manager.should_stop_loss(position.entry_price, current_price, position.position_type) or
                self.risk_manager.should_take_profit(position.entry_price, current_price, position.position_type)):
                positions_to_close.append(symbol)
        
        for symbol in positions_to_close:
            self.close_position(symbol)
    
    def close_position(self, symbol: str):
        """Close a specific position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            current_price = self.data_provider.get_real_time_price(symbol)
            self.execute_trade(symbol, 'sell', position.quantity, current_price)
    
    def close_all_positions(self):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            self.close_position(symbol)
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            current_price = self.data_provider.get_real_time_price(symbol)
            position_value = position.quantity * current_price
            total_value += position_value
        
        return total_value
    
    def run_live_trading(self, symbols: List[str], check_interval: int = 300):
        """Run live trading system"""
        logger.info(f"Starting live trading for symbols: {symbols}")
        
        while True:
            try:
                for symbol in symbols:
                    df = self.data_provider.get_historical_data(symbol, period='6m')
                    if df.empty:
                        continue
                    
                    df_with_signals = self.strategy.generate_signals(df)
                    latest_signal = df_with_signals['signal'].iloc[-1]
                    current_price = self.data_provider.get_real_time_price(symbol)
                    
                    if latest_signal == 1 and symbol not in self.positions:
                        quantity = int((self.current_capital * 0.1) / current_price)  # Use 10% of capital
                        if quantity > 0:
                            self.execute_trade(symbol, 'buy', quantity, current_price)
                    
                    elif latest_signal == -1 and symbol in self.positions:
                        self.close_position(symbol)
                
                self.check_risk_management()
                
                portfolio_value = self.get_portfolio_value()
                logger.info(f"Portfolio Value: ${portfolio_value:,.2f}, Positions: {len(self.positions)}")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Trading system stopped by user")
                self.close_all_positions()
                break
            except Exception as e:
                logger.error(f"Error in live trading: {e}")
                time.sleep(60)  

# Example usage and demo
def demo_trading_system():
    """Demo function to show how to use the trading system"""
    
    API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
    
    data_provider = AlphaVantageProvider(API_KEY)
    strategy = MomentumStrategy(short_window=20, long_window=50)
    risk_manager = RiskManager(max_position_size=0.1, stop_loss=0.05, take_profit=0.15)
    backtester = Backtester(initial_capital=100000)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("=== ALGORITHMIC TRADING SYSTEM DEMO ===\n")
    
    for symbol in symbols[:2]:  
        print(f"\n--- Backtesting {symbol} ---")
        df = data_provider.get_historical_data(symbol, period='1y')
        
        if not df.empty:
            results = backtester.run_backtest(df, strategy)
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Annualized Return: {results['annualized_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        else:
            print(f"No data available for {symbol}")
    
    trading_system = TradingSystem(data_provider, strategy, risk_manager, initial_capital=100000)
    
    print(f"\n--- Trading System Initialized ---")
    print(f"Initial Capital: ${trading_system.initial_capital:,.2f}")
    print(f"Current Capital: ${trading_system.current_capital:,.2f}")
    print(f"Positions: {len(trading_system.positions)}")
    
    trading_system.run_live_trading(symbols, check_interval=300)

if __name__ == "__main__":
    demo_trading_system()