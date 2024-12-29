import ccxt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
import time


class MovingAverageCrossoverML:
    def __init__(self, config):
        """
        Initialize the Moving Average Crossover ML strategy class.

        Parameters:
            config (dict): Configuration dictionary with strategy parameters.
        """
        # Strategy Configuration
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.short_window = config.get('short_window', 10)
        self.long_window = config.get('long_window', 50)
        self.take_profit = config.get('take_profit', 0.02)  # 2% take profit
        self.stop_loss = config.get('stop_loss', 0.01)      # 1% stop loss
        self.leverage = config.get('leverage', 5)
        self.investment_amount = config.get('investment_amount', 50)  # USD
        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.exchange = self._initialize_exchange()

        # ML Model and Scaler
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

        # Track open orders
        self.open_orders = []

        # Logging Setup
        self.log_file = config.get('log_file', 'trade_log.csv')
        self._setup_logging()

    def _initialize_exchange(self):
        """
        Initialize the Bybit Demo Account using CCXT.
        """
        exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            },
            'urls': {
                'api': 'https://api-testnet.bybit.com'  # Bybit testnet API
            }
        })
        print(f"Initialized Bybit Demo Account for {self.symbol}")
        return exchange

    def _setup_logging(self):
        """
        Setup trade logging to a CSV file.
        """
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as file:
                file.write('timestamp,symbol,side,price,amount,profit\n')
        print(f"Logging trades to {self.log_file}")

    def fetch_historical_data_yfinance(self, period='1mo', interval='1h'):
        """
        Fetch historical price data using Yahoo Finance.

        Parameters:
            period (str): Period of data to fetch (e.g., '60d').
            interval (str): Data interval (e.g., '1h').

        Returns:
            pd.DataFrame: Dataframe containing price data with datetime index.
        """
        print(f"Fetching historical data for {self.symbol} using yfinance...")
        symbol_yf = self.symbol.replace("/", "")
        data = yf.download(tickers=symbol_yf, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data fetched. Check the symbol and parameters.")
        data = data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        return data

    def preprocess_data(self, df):
        """
        Calculate moving averages and generate trading signals.
        """
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        df['signal'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
        return df.dropna()

    def train_model(self, df):
        """
        Train the machine learning model.
        """
        X = df[['short_ma', 'long_ma']]
        y = df['signal']
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, self.model.predict(X_test))
        print(f"Model accuracy: {accuracy:.2f}")

    def predict_signal(self, df):
        """
        Predict the next trading signal.
        """
        latest_data = df[['short_ma', 'long_ma']].iloc[-1:].values
        scaled_data = self.scaler.transform(latest_data)
        signal = self.model.predict(scaled_data)[0]
        return signal

    def place_order(self, signal):
        """
        Place a futures market order based on the predicted signal.
        """
        side = 'buy' if signal == 1 else 'sell'
        try:
            # Fetch ticker price
            print(f"11111111111")  # Debugging
        
            ticker = self.exchange.fetch_ticker('BTCUSDT')
            print(f"22222222222")  # Debugging
            print(f"Ticker response: {ticker}")  # Debugging
            price = ticker['last']

            # Calculate order size
            amount = (self.investment_amount * self.leverage) / price
            stop_loss = price * (1 - self.stop_loss) if side == 'buy' else price * (1 + self.stop_loss)
            take_profit = price * (1 + self.take_profit) if side == 'buy' else price * (1 - self.take_profit)

            print(f"price: {price}")  # Debugging
            print(f"amount: {amount}")  # Debugging
            print(f"stop_loss: {stop_loss}")  # Debugging
            print(f"take_profit: {take_profit}")  # Debugging

            # Place market order
            order = self.exchange.create_market_order(self.symbol, side, amount)
            print(f"Order response: {order}")  # Debugging full order response

            # Append to open orders if successful
            self.open_orders.append({
                'side': side,
                'price': price,
                'amount': amount,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            print(f"Placed {side.upper()} order at {price}")
        except Exception as e:
            print(f"Error placing order: {e}")




    def monitor_trades(self):
        """
        Monitor open trades for stop loss or take profit conditions.
        """
        for trade in self.open_orders[:]:
            current_price = self.exchange.fetch_ticker(self.symbol)['last']
            if (trade['side'] == 'buy' and (current_price <= trade['stop_loss'] or current_price >= trade['take_profit'])) or \
               (trade['side'] == 'sell' and (current_price >= trade['stop_loss'] or current_price <= trade['take_profit'])):
                self.close_order(trade, current_price)

    def close_order(self, trade, close_price):
        """
        Close an open trade.
        """
        side = 'sell' if trade['side'] == 'buy' else 'buy'
        profit = (close_price - trade['price']) * trade['amount'] if trade['side'] == 'buy' else (trade['price'] - close_price) * trade['amount']
        self.exchange.create_market_order(self.symbol, side, trade['amount'])
        self.log_trade(datetime.now(), self.symbol, side, close_price, trade['amount'], round(profit, 2))
        self.open_orders.remove(trade)

    def log_trade(self, timestamp, symbol, side, price, amount, profit):
        """
        Log trade details to a CSV file.
        """
        with open(self.log_file, 'a') as file:
            file.write(f"{timestamp},{symbol},{side},{price},{amount},{profit}\n")
        print(f"Logged {side.upper()} trade at {price} with profit {profit}")

    def sleep_with_details(self, duration):
        """
        Provide detailed sleep updates.
        """
        print(f"Sleeping for {duration // 60} minutes...")
        for i in range(duration // 60):
            time.sleep(60)
            print(f"Time passed: {i + 1} minute(s)")
        print("Sleep complete.")

    def run_strategy(self, sleep_interval=300):
        """
        Run the strategy in an infinite loop.
        """
        print("Starting the Moving Average Crossover ML Strategy...")
        while True:
            try:
                df = self.fetch_historical_data_yfinance()
                df = self.preprocess_data(df)
                self.train_model(df)
                signal = self.predict_signal(df)
                print(f"Predicted Signal: {'BUY' if signal == 1 else 'SELL'}")
                self.place_order(signal)
                self.monitor_trades()
                self.sleep_with_details(sleep_interval)
            except Exception as e:
                print(f"Error: {e}")
                self.sleep_with_details(sleep_interval)


# Example configuration for Bybit demo account
config = {
    'symbol': 'BTC-USD',
    'short_window': 10,
    'long_window': 50,
    'take_profit': 0.02,
    'stop_loss': 0.01,
    'leverage': 5,
    'investment_amount': 50,
    'api_key': 'qodKyG2dr2xYd7rZOM',
    'api_secret': 'rKKJGynMHJDIVoSpGAmqwUD1eB7LdK9sd8eQ',
    'log_file': 'trade_log.csv'
}

# Run the strategy
if __name__ == "__main__":
    strategy = MovingAverageCrossoverML(config)
    strategy.run_strategy(sleep_interval=300)
