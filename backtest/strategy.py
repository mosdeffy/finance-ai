"""
strategy.py
Backtrader strategy using LSTM predictions to generate buy/sell signals.
Reports Sharpe ratio, max drawdown, and total return.
"""
import backtrader as bt
import numpy as np
import pandas as pd


class LSTMStrategy(bt.Strategy):
    """
    Executes long/flat positions based on LSTM directional predictions.
    Buy when predicted probability >= threshold; sell when below.
    """
    params = dict(threshold=0.5)

    def __init__(self):
        self.predictions = self.datas[1].close
        self.order = None

    def next(self):
        if self.order:
            return
        signal = self.predictions[0]
        if not self.position:
            if signal >= self.p.threshold:
                self.order = self.buy()
        else:
            if signal < self.p.threshold:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None


def run_backtest(price_df: pd.DataFrame, predictions: np.ndarray, threshold: float = 0.5):
    """
    Run a full backtest with LSTM predictions as signals.
    Args:
        price_df: OHLCV DataFrame with DatetimeIndex
        predictions: Array of LSTM output probabilities
        threshold: Decision boundary for buy signal (default 0.5)
    """
    cerebro = bt.Cerebro()
    cerebro.addstrategy(LSTMStrategy, threshold=threshold)
    data_feed = bt.feeds.PandasData(dataname=price_df)
    cerebro.adddata(data_feed)
    pred_df = pd.DataFrame(
        {"close": predictions},
        index=price_df.index[-len(predictions):]
    )
    pred_feed = bt.feeds.PandasData(
        dataname=pred_df,
        openinterest=None, open=None, high=None, low=None, volume=None
    )
    cerebro.adddata(pred_feed)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.04, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    print("Starting Portfolio Value: $100,000.00")
    results = cerebro.run()
    strat = results[0]
    final = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    max_dd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown")
    total_ret = strat.analyzers.returns.get_analysis().get("rtot")
    print("\n========== Backtest Results ==========")
    print(f"Final Portfolio Value : {final:>12,.2f}")
    print(f"Total Return         : {total_ret:.2%}" if isinstance(total_ret, float) else "Total Return: N/A")
    print(f"Sharpe Ratio         : {sharpe:.4f}" if isinstance(sharpe, float) else "Sharpe Ratio: N/A")
    print(f"Max Drawdown         : {max_dd:.2f}%" if isinstance(max_dd, float) else "Max Drawdown: N/A")
    print("======================================")
    cerebro.plot(style="candlestick", volume=False)


if __name__ == "__main__":
    print("Usage: import strategy; strategy.run_backtest(price_df, predictions, threshold=0.52)")
