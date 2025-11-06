# Indian Stock Market Configuration for FinRL
from __future__ import annotations

# Indian Market Timezone
INDIAN_TIMEZONE = "Asia/Kolkata"

# Trading Hours (NSE/BSE)
INDIAN_MARKET_HOURS = {
    'market_open': '09:15',
    'market_close': '15:30',
    'pre_market_open': '09:00',
    'pre_market_close': '09:15',
    'post_market_open': '15:40', 
    'post_market_close': '16:00'
}

# Indian Stock API Configuration
INDIAN_API_CONFIG = {
    'base_url': 'https://stock.indianapi.in',
    'timeout': 30,
    'max_retries': 3,
    'rate_limit': 100  # requests per minute
}

# Popular Indian Stock Lists

# NIFTY 50 - Top 50 companies by market cap
NIFTY_50_TICKERS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
    'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 
    'LT', 'ASIANPAINT', 'AXISBANK', 'MARUTI', 'SUNPHARMA',
    'ULTRACEMCO', 'BAJFINANCE', 'NESTLEIND', 'TITAN', 'WIPRO',
    'ONGC', 'NTPC', 'POWERGRID', 'HCLTECH', 'COALINDIA',
    'BAJAJFINSV', 'GRASIM', 'TECHM', 'HINDALCO', 'INDUSINDBK',
    'ADANIPORTS', 'HEROMOTOCO', 'TATAMOTORS', 'CIPLA', 'EICHERMOT',
    'UPL', 'JSWSTEEL', 'BRITANNIA', 'DRREDDY', 'APOLLOHOSP',
    'DIVISLAB', 'BAJAJ-AUTO', 'SHREECEM', 'TATACONSUM', 'IOC',
    'TATASTEEL', 'ADANIENT', 'SBILIFE', 'HDFCLIFE', 'BPCL'
]

# NIFTY Bank - Banking sector stocks
NIFTY_BANK_TICKERS = [
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK',
    'INDUSINDBK', 'AUBANK', 'BANDHANBNK', 'FEDERALBNK', 'IDFCFIRSTB',
    'PNB', 'BANKBARODA'
]

# NIFTY IT - Information Technology sector
NIFTY_IT_TICKERS = [
    'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 
    'LTI', 'MINDTREE', 'MPHASIS', 'COFORGE', 'LTTS'
]

# NIFTY Auto - Automobile sector 
NIFTY_AUTO_TICKERS = [
    'MARUTI', 'TATAMOTORS', 'HEROMOTOCO', 'BAJAJ-AUTO', 'EICHERMOT',
    'M&M', 'ASHOKLEY', 'BHARATFORG', 'MOTHERSON', 'BOSCHLTD'
]

# Popular mid-cap stocks for diversified portfolio
INDIAN_MIDCAP_TICKERS = [
    'GODREJCP', 'PIDILITIND', 'MCDOWELL-N', 'DABUR', 'MARICO',
    'COLPAL', 'BERGEPAINT', 'PAGEIND', 'CONCOR', 'TORNTPHARM',
    'LUPIN', 'BIOCON', 'CADILAHC', 'ALKEM', 'AUROPHARMA'
]

# Complete list for comprehensive testing (Top 100)
TOP_100_INDIAN_STOCKS = NIFTY_50_TICKERS + [
    'ADANIGREEN', 'ADANITRANS', 'ACC', 'AMARAJABAT', 'AMBUJACEM',
    'APOLLOTYRE', 'ASHOKLEY', 'AUBANK', 'AUROPHARMA', 'BALKRISIND',
    'BANDHANBNK', 'BATAINDIA', 'BEL', 'BERGEPAINT', 'BHARATFORG',
    'BIOCON', 'BOSCHLTD', 'CADILAHC', 'CANBK', 'CHOLAFIN',
    'COLPAL', 'CONCOR', 'CUMMINSIND', 'DABUR', 'DELTACORP',
    'DLF', 'FEDERALBNK', 'GAIL', 'GODREJCP', 'HAVELLS',
    'HDFC', 'IDFCFIRSTB', 'INDIANB', 'JUBLFOOD', 'L&TFH',
    'LICHSGFIN', 'LTI', 'LUPIN', 'M&M', 'MARICO',
    'MCDOWELL-N', 'MINDTREE', 'MOTHERSON', 'MPHASIS', 'MRF',
    'NAUKRI', 'NMDC', 'OBEROIRLTY', 'PAGEIND', 'PETRONET'
]

# Market Indices for benchmarking
INDIAN_INDICES = {
    'NIFTY50': 'NSEI',
    'BANKNIFTY': 'NSEBANK', 
    'NIFTYIT': 'NSIEIT',
    'SENSEX': 'BSE30',
    'NIFTYMIDCAP': 'NSMIDCP',
    'NIFTYSMALLCAP': 'NSSMCP'
}

# Sector-wise classification
SECTOR_CLASSIFICATION = {
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK'],
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTI', 'MINDTREE'],
    'Auto': ['MARUTI', 'TATAMOTORS', 'HEROMOTOCO', 'BAJAJ-AUTO', 'EICHERMOT', 'M&M'],
    'Pharma': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'LUPIN', 'BIOCON', 'DIVISLAB'],
    'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO'],
    'Telecom': ['BHARTIARTL', 'IDEA', 'RCOM'],
    'Oil_Gas': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL'],
    'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'NMDC'],
    'Cement': ['ULTRACEMCO', 'ACC', 'AMBUJACEM', 'SHREECEM', 'RAMCOCEM'],
    'Realty': ['DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'BRIGADE']
}

# Risk categories for portfolio management  
RISK_CATEGORIES = {
    'Low_Risk': ['HDFCBANK', 'ICICIBANK', 'TCS', 'INFY', 'HINDUNILVR', 'NESTLEIND'],
    'Medium_Risk': ['RELIANCE', 'LT', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'BAJFINANCE'],
    'High_Risk': ['TATASTEEL', 'JSWSTEEL', 'TATAMOTORS', 'YESBANK', 'IDEA', 'RCOM']
}

# Default portfolio for RL training (balanced across sectors)
DEFAULT_INDIAN_PORTFOLIO = [
    # Banking (20%)
    'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK',
    
    # IT (15%) 
    'TCS', 'INFY', 'WIPRO',
    
    # FMCG (10%)
    'HINDUNILVR', 'ITC',
    
    # Auto (10%)
    'MARUTI', 'TATAMOTORS',
    
    # Pharma (10%)
    'SUNPHARMA', 'CIPLA',
    
    # Oil & Gas (10%)
    'RELIANCE', 'ONGC',
    
    # Metals (10%)
    'TATASTEEL', 'JSWSTEEL',
    
    # Infrastructure (10%)
    'LT', 'NTPC',
    
    # Telecom (5%)
    'BHARTIARTL'
]

# Time periods for historical data  
INDIAN_TIME_PERIODS = {
    '1m': '1 Month',
    '6m': '6 Months', 
    '1yr': '1 Year',
    '3yr': '3 Years',
    '5yr': '5 Years',
    '10yr': '10 Years',
    'max': 'Maximum Available'
}

# Technical indicators commonly used for Indian markets
INDIAN_TECHNICAL_INDICATORS = [
    'macd',      # MACD
    'rsi_14',    # RSI (14 periods)
    'rsi_30',    # RSI (30 periods) 
    'boll_ub',   # Bollinger Upper Band
    'boll_lb',   # Bollinger Lower Band
    'cci_30',    # CCI (30 periods)
    'dx_30',     # Directional Index
    'close_30_sma',  # 30-day Simple Moving Average
    'close_60_sma',  # 60-day Simple Moving Average
    'wr_14',     # Williams %R
    'atr_14',    # Average True Range
    'adx_14',    # ADX
    'obv',       # On-Balance Volume
    'vwap'       # Volume Weighted Average Price
]

# Market holidays (major ones - extend as needed)
INDIAN_MARKET_HOLIDAYS_2025 = [
    '2025-01-26',  # Republic Day
    '2025-03-14',  # Holi
    '2025-04-18',  # Good Friday  
    '2025-05-01',  # Maharashtra Day
    '2025-08-15',  # Independence Day
    '2025-10-02',  # Gandhi Jayanti
    '2025-10-24',  # Dussehra
    '2025-11-12',  # Diwali (Lakshmi Puja)
    '2025-11-13',  # Diwali Balipratipada
    '2025-12-25'   # Christmas
]

# Currency and market info
INDIAN_MARKET_INFO = {
    'currency': 'INR',
    'decimal_places': 2,
    'lot_size': 1,  # Most stocks trade in multiples of 1
    'tick_size': 0.05,  # Minimum price movement
    'settlement': 'T+2',  # Settlement cycle
    'trading_days': 'Monday to Friday'
}