# Automated Trading System

A comprehensive algorithmic trading system implementing momentum-based strategies with advanced risk management, backtesting capabilities, and real-time market data integration.

## 🚀 Features

### Core Trading Algorithm
- **Momentum-Based Strategy**: Uses multiple technical indicators for signal generation
- **Multi-Indicator Analysis**: SMA crossovers, RSI, MACD, and Bollinger Bands
- **Real-Time Execution**: Live trading with configurable check intervals
- **Position Management**: Automated entry and exit based on technical signals

### Risk Management
- **Position Sizing**: Maximum 10% of portfolio per position
- **Stop Loss**: Configurable stop-loss orders (default: 5%)
- **Take Profit**: Automated profit-taking (default: 15%)
- **Drawdown Protection**: Maximum portfolio drawdown limits (default: 15%)
- **Portfolio Monitoring**: Real-time risk assessment and position adjustment

### Backtesting Engine
- **Historical Analysis**: Comprehensive backtesting on historical data
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate, and more
- **Return Analysis**: Total returns, annualized returns, and volatility metrics
- **Trade Statistics**: Detailed trade-by-trade analysis

### Data Integration
- **Alpha Vantage API**: Real-time and historical market data
- **Multiple Timeframes**: Support for various data periods (1y, 6m, etc.)
- **Error Handling**: Robust API error handling and retry mechanisms
- **Rate Limiting**: Built-in API rate limit management

## 📋 Requirements

### Python Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
requests>=2.25.0
scikit-learn>=1.0.0
```

### External APIs
- **Alpha Vantage API Key** (free tier available)
- Internet connection for real-time data

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/automated-trading-system.git
   cd automated-trading-system
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy requests scikit-learn
   ```

3. **Get Alpha Vantage API Key**
   - Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Sign up for a free API key
   - Note: Free tier allows 5 API requests per minute, 500 requests per day

4. **Configure API Key**
   ```python
   # In the demo_trading_system() function
   API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY_HERE"
   ```

## 🚀 Quick Start

### Basic Usage
```python
from trading_system import *

# Initialize components
data_provider = AlphaVantageProvider("YOUR_API_KEY")
strategy = MomentumStrategy(short_window=20, long_window=50)
risk_manager = RiskManager(max_position_size=0.1, stop_loss=0.05)

# Run backtest
backtester = Backtester(initial_capital=100000)
df = data_provider.get_historical_data('AAPL', period='1y')
results = backtester.run_backtest(df, strategy)
backtester.print_results()
```

### Running the Demo
```bash
python trading_system.py
```

## 📊 Strategy Details

### Momentum Strategy Indicators

**Moving Averages**
- Short-term: 20-day Simple Moving Average
- Long-term: 50-day Simple Moving Average
- Signal: Buy when short MA > long MA

**RSI (Relative Strength Index)**
- Period: 14 days
- Overbought: RSI > 70 (avoid buying)
- Oversold: RSI < 30 (avoid selling)

**MACD (Moving Average Convergence Divergence)**
- Fast EMA: 12 days
- Slow EMA: 26 days
- Signal Line: 9-day EMA of MACD
- Signal: Buy when MACD > Signal Line

**Bollinger Bands**
- Period: 20 days
- Standard Deviations: 2
- Signal: Confirm trend direction

### Buy Signal Criteria
All conditions must be met:
1. Short MA > Long MA (uptrend)
2. RSI < 70 (not overbought)
3. MACD > Signal Line (momentum positive)
4. Price > Lower Bollinger Band (trend confirmation)

### Sell Signal Criteria
All conditions must be met:
1. Short MA < Long MA (downtrend)
2. RSI > 30 (not oversold)
3. MACD < Signal Line (momentum negative)
4. Price < Upper Bollinger Band (trend confirmation)

## ⚙️ Configuration

### Strategy Parameters
```python
strategy = MomentumStrategy(
    short_window=20,    # Short MA period
    long_window=50,     # Long MA period
    rsi_period=14       # RSI calculation period
)
```

### Risk Management Settings
```python
risk_manager = RiskManager(
    max_position_size=0.1,  # 10% max position size
    max_drawdown=0.15,      # 15% max portfolio drawdown
    stop_loss=0.05,         # 5% stop loss
    take_profit=0.15        # 15% take profit
)
```

### Live Trading Configuration
```python
trading_system.run_live_trading(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    check_interval=300  # Check every 5 minutes
)
```

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

- **Total Return**: Overall portfolio return
- **Annualized Return**: Year-over-year return rate
- **Volatility**: Portfolio risk measurement
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Trade Count**: Total number of executed trades

## 🗄️ Database Schema

### Trades Table
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    action TEXT,        -- 'buy' or 'sell'
    quantity REAL,
    price REAL,
    timestamp TEXT,
    pnl REAL           -- Profit/Loss for the trade
);
```

### Positions Table
```sql
CREATE TABLE positions (
    symbol TEXT PRIMARY KEY,
    quantity REAL,
    entry_price REAL,
    entry_time TEXT,
    position_type TEXT  -- 'long' or 'short'
);
```

## 🔧 Advanced Usage

### Custom Data Provider
```python
class CustomDataProvider(DataProvider):
    def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        # Implement your data source
        pass
    
    def get_real_time_price(self, symbol: str) -> float:
        # Implement real-time price fetching
        pass
```

### Custom Strategy
```python
class CustomStrategy(MomentumStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implement your custom strategy logic
        df = super().generate_signals(df)
        # Add custom modifications
        return df
```

## 📋 API Rate Limits

### Alpha Vantage Limits
- **Free Tier**: 5 requests/minute, 500 requests/day
- **Premium**: Higher limits available

### Optimization Tips
- Cache historical data to reduce API calls
- Use appropriate check intervals for live trading
- Implement exponential backoff for rate limit handling

## 🛡️ Risk Warnings

**Important Disclaimers:**

⚠️ **Trading Risks**
- All trading involves substantial risk of loss
- Past performance does not guarantee future results
- Only trade with capital you can afford to lose

⚠️**System Risks**
- API failures can interrupt trading
- Network issues may cause missed opportunities
- System bugs could result in unintended trades

⚠️ **Regulatory Compliance**
- Ensure compliance with local trading regulations
- Some jurisdictions may require licensing for automated trading
- Tax implications vary by location

## 🔄 Testing and Validation

### Backtesting Workflow
1. **Data Validation**: Verify historical data quality
2. **Strategy Testing**: Test on multiple time periods
3. **Parameter Optimization**: Fine-tune strategy parameters
4. **Risk Assessment**: Validate risk management rules
5. **Paper Trading**: Test with virtual money first

### Recommended Testing Process
```python
# Test multiple symbols and time periods
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
periods = ['1y', '6m', '3m']

for symbol in symbols:
    for period in periods:
        df = data_provider.get_historical_data(symbol, period)
        results = backtester.run_backtest(df, strategy)
        # Analyze results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-strategy`)
3. Commit your changes (`git commit -am 'Add new momentum strategy'`)
4. Push to the branch (`git push origin feature/new-strategy`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

### Common Issues

**API Key Errors**
- Verify API key is correct and active
- Check API usage limits
- Ensure internet connection is stable

**Data Issues**
- Validate symbol names (use correct ticker symbols)
- Check data availability for requested periods
- Verify market hours for real-time data

**Database Errors**
- Ensure write permissions in project directory
- Check SQLite installation
- Verify database file isn't corrupted

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check inline code documentation
- **Alpha Vantage Support**: For API-related issues

## 🔄 Version History

- **v1.0.0**: Initial release with momentum strategy and backtesting
- **v1.1.0**: Added risk management and live trading capabilities
- **v1.2.0**: Enhanced database logging and error handling

## 🎯 Roadmap

### Planned Features
- [ ] Machine learning-based strategy optimization
- [ ] Multiple data provider support (Yahoo Finance, IEX, etc.)
- [ ] Web dashboard for monitoring
- [ ] Advanced order types (limit orders, trailing stops)
- [ ] Portfolio optimization algorithms
- [ ] Cryptocurrency trading support
- [ ] Cloud deployment options
- [ ] Email/SMS alert system

### Performance Targets
- Target: 15%+ annualized returns
- Maximum drawdown: <15%
- Sharpe ratio: >1.0
- Win rate: >55%

---

**Disclaimer**: This software is for educational and research purposes only. The authors are not responsible for any financial losses incurred through the use of this system. Always conduct thorough testing and consider consulting with a financial advisor before deploying any automated trading system with real money.