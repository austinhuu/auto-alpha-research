import time
import os
import argparse
from pathlib import Path

import pandas as pd


def download_tushare_daily_data(
    token,
    symbols,
    start_date,
    end_date,
    output_path,
    sleep_seconds=0.25,
):
    """
    使用 Tushare 下载 A 股日线数据，并转换成项目需要的格式。

    项目需要字段：
    datetime, symbol, open, high, low, close, volume, amount

    Tushare daily 原始字段：
    ts_code, trade_date, open, high, low, close, vol, amount

    注意：
    - Tushare 的 vol 通常是成交量，单位为手；
    - amount 通常是成交额，单位为千元；
    - 如果你后面只做因子相对比较，单位问题影响不大；
    - 如果做真实成交额金额解释，需要统一单位。
    """

    import tushare as ts

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    token = token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Set TUSHARE_TOKEN in your environment before downloading data.")
    ts.set_token(token)
    pro = ts.pro_api()

    all_data = []

    for symbol in symbols:
        print(f"Downloading {symbol} ...")

        df = pro.daily(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        if df is None or len(df) == 0:
            print(f"Warning: no data for {symbol}")
            continue

        df = df.copy()

        df = df.rename(
            columns={
                "trade_date": "datetime",
                "ts_code": "symbol",
                "vol": "volume",
            }
        )

        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d")

        keep_cols = [
            "datetime",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]

        df = df[keep_cols]

        all_data.append(df)

        time.sleep(sleep_seconds)

    if len(all_data) == 0:
        raise RuntimeError("No data downloaded. Please check token, symbols, and date range.")

    result = pd.concat(all_data, ignore_index=True)

    result = result.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved data to {output_path}")
    print(f"Shape: {result.shape}")
    print(f"Symbols: {result['symbol'].nunique()}")
    print(f"Date range: {result['datetime'].min()} -> {result['datetime'].max()}")

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download authorized A-share daily data using your Tushare account")
    parser.add_argument("--symbols", nargs="+", required=True, help="E.g. 000001.SZ 600519.SH")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", required=True, help="CSV file path (not committed to GitHub)")
    args = parser.parse_args()
    download_tushare_daily_data(
        token=os.getenv("TUSHARE_TOKEN"), symbols=args.symbols,
        start_date=args.start, end_date=args.end, output_path=args.output,
    )
