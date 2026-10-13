from pathlib import Path
from datetime import datetime

import pandas as pd

from vnpy_ctastrategy import (
    CtaTemplate,
    BarData,
    TickData,
    OrderData,
    TradeData,
    StopOrder,
)


class AlphaSignalStrategy(CtaTemplate):
    """
    V6：Alpha 信号驱动策略。

    逻辑：
    1. 离线系统生成 alpha_signals.csv；
    2. VeighNa 策略读取该文件；
    3. 根据当前 symbol + date 匹配 signal；
    4. signal = 1 时持有多头；
    5. signal = -1 或 0 时空仓。
    """

    author = "Austin Hu"

    signal_file = "alpha_signals.csv"
    fixed_size = 100
    use_date_only = True
    enable_live_orders = False  # DANGEROUS opt-in; broker/risk/symbol mapping unvalidated

    parameters = [
        "signal_file",
        "fixed_size",
        "use_date_only",
        "enable_live_orders",
    ]

    variables = [
        "signal",
        "alpha_score",
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.signal = 0
        self.alpha_score = 0.0

        self.signal_df = None
        self.signal_map = {}

    def on_init(self):
        self.write_log("AlphaSignalStrategy 初始化开始")
        self.load_signals()
        self.load_bar(10)

    def on_start(self):
        self.write_log("AlphaSignalStrategy 启动")

    def on_stop(self):
        self.write_log("AlphaSignalStrategy 停止")

    def load_signals(self):
        path = Path(self.signal_file)

        if not path.exists():
            self.write_log(f"信号文件不存在: {path}")
            return

        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])

        required_cols = ["datetime", "symbol", "alpha_score", "signal"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            self.write_log(f"信号文件缺少字段: {missing}")
            return

        self.signal_df = df

        for _, row in df.iterrows():
            symbol = str(row["symbol"])

            if self.use_date_only:
                dt_key = row["datetime"].date()
            else:
                dt_key = row["datetime"]

            key = (symbol, dt_key)

            self.signal_map[key] = {
                "signal": int(row["signal"]),
                "alpha_score": float(row["alpha_score"]) if pd.notna(row["alpha_score"]) else 0.0,
            }

        self.write_log(f"信号文件加载成功，共 {len(self.signal_map)} 条记录")

    def on_tick(self, tick: TickData):
        pass

    def on_bar(self, bar: BarData):
        # Research adapter defaults to monitor-only; do not submit/cancel orders.
        if self.enable_live_orders:
            self.cancel_all()

        current_symbol = bar.symbol
        # CSV usually stores ts_code (e.g. 000001.SZ), while bar.symbol might not.

        if self.use_date_only:
            dt_key = bar.datetime.date()
        else:
            dt_key = bar.datetime

        # EOD signal for date t is only available after that day's close;
        # never use same-date signal on an intraday bar.
        if self.use_date_only:
            eligible = [key for key in self.signal_map
                        if key[0] == current_symbol and key[1] < dt_key]
            if not eligible:
                return
            key = max(eligible, key=lambda item: item[1])
        else:
            key = (current_symbol, dt_key)

        info = self.signal_map.get(key)

        if info is None:
            self.write_log(f"未找到信号: {current_symbol}, {dt_key}")
            return

        self.signal = info["signal"]
        self.alpha_score = info["alpha_score"]

        self.write_log(
            f"当前信号: symbol={current_symbol}, date={dt_key}, "
            f"signal={self.signal}, alpha_score={self.alpha_score:.6f}, pos={self.pos}"
        )

        if not self.enable_live_orders:
            self.write_log("Monitor-only: signal observed, no orders submitted")
            self.put_event()
            return

        if self.signal == 1:
            if self.pos <= 0:
                if self.pos < 0:
                    self.cover(bar.close_price * 1.01, abs(self.pos))

                self.buy(bar.close_price * 1.01, self.fixed_size)

        elif self.signal <= 0:
            if self.pos > 0:
                self.sell(bar.close_price * 0.99, abs(self.pos))

        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.write_log(f"成交: {trade.direction}, {trade.volume}, {trade.price}")
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        pass