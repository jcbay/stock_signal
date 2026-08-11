"""
数据获取模块 v2 - 新浪 + 腾讯财经API
扩展功能:
1. 多周期K线数据 (日/周/月)
2. 更多基本面字段 (ROE/营收增速等)
3. 大盘指数数据获取 (上证/深证)
4. 数据缓存到 SQLite (db.py)
"""

import subprocess
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from db import cache_kline, load_kline, kline_cache_age, cache_fundamentals, load_fundamentals


def is_etf(stock_code):
    """判断是否为ETF/LOF基金

    上海ETF: 510xxx-518xxx, 588xxx, 562xxx-563xxx
    深圳ETF: 159xxx
    LOF: 501xxx(上海), 161xxx-167xxx(深圳)
    """
    prefixes = ("51", "56", "58", "50", "15", "16")
    return stock_code.startswith(prefixes)


def get_symbol_prefix(stock_code):
    """根据代码判断市场前缀

    上海(sh): 6开头(主板), 5开头(ETF/基金), 9开头(B股)
    深圳(sz): 0开头(主板), 3开头(创业板), 1开头(基金/ETF), 2开头(B股)
    """
    if stock_code.startswith(("6", "5", "9")):
        return "sh"
    else:
        return "sz"


def get_market_id(stock_code):
    """市场ID: 上海=1, 深圳=0"""
    return 1 if stock_code.startswith(("6", "5", "9")) else 0


def curl_get(url, timeout=20, encoding="utf-8", extra_headers=None):
    """用系统 curl 获取数据，绕过 Python HTTP 库网络限制"""
    cmd = ["curl", "-s", "--noproxy", "*", "-m", str(timeout)]
    if extra_headers:
        for k, v in extra_headers.items():
            cmd.extend(["-H", f"{k}: {v}"])
    cmd.append(url)

    env = dict(os.environ)
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        env.pop(k, None)
    env["NO_PROXY"] = "*"

    result = subprocess.run(cmd, capture_output=True, env=env)
    if result.returncode != 0 or not result.stdout:
        stderr_text = result.stderr.decode(encoding, errors="replace") if result.stderr else ""
        raise ValueError(f"curl 请求失败 (rc={result.returncode}): {stderr_text or 'empty response'}")

    return result.stdout.decode(encoding, errors="replace")


def fetch_hist_data(stock_code, period="3m", scale=240, use_cache=True):
    """获取历史K线数据 (新浪API)，返回 DataFrame

    scale参数:
    240 = 日K线
    1200 = 周K线 (5天)
    7200 = 月K线 (30天)

    period参数: 1m=30条, 3m=90条, 6m=180条, 1y=365条
    """
    prefix = get_symbol_prefix(stock_code)
    symbol = f"{prefix}{stock_code}"

    datalen_map = {"1m": 30, "3m": 90, "6m": 180, "1y": 365}
    datalen = datalen_map.get(period, 90)

    # 缓存检查
    if use_cache:
        cache_latest = kline_cache_age(stock_code, "daily")
        if cache_latest:
            # 如果缓存最新日期是今天或昨天，直接用缓存
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            if cache_latest >= yesterday:
                cached_df = load_kline(stock_code, "daily", days=datalen + 10)
                if len(cached_df) >= datalen * 0.8:  # 缓存数据足够
                    return cached_df.iloc[-datalen:].reset_index(drop=True)

    url = (
        f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale={scale}&ma=no&datalen={datalen}"
    )

    raw = curl_get(url, extra_headers={"Referer": "https://finance.sina.com.cn"})
    data = json.loads(raw)

    if not data or len(data) < 10:
        raise ValueError(f"股票 {stock_code} 历史数据不足")

    rows = []
    for item in data:
        rows.append({
            "日期": item["day"],
            "开盘": float(item["open"]),
            "收盘": float(item["close"]),
            "最高": float(item["high"]),
            "最低": float(item["low"]),
            "成交量": float(item["volume"]),
        })

    df = pd.DataFrame(rows)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)

    # 计算衍生字段
    df["成交额"] = df["成交量"] * 100 * (df["开盘"] + df["收盘"]) / 2
    df["涨跌额"] = df["收盘"].diff().fillna(0)
    df["涨跌幅"] = df["收盘"].pct_change().fillna(0) * 100
    df["振幅"] = ((df["最高"] - df["最低"]) / df["收盘"].shift(1) * 100).fillna(0)
    df["换手率"] = 0  # 新浪API不提供换手率

    # 缓存到数据库
    if use_cache and scale == 240:
        cache_kline(stock_code, df, "daily")

    return df


def fetch_weekly_data(stock_code, datalen=52, use_cache=True):
    """获取周K线数据"""
    prefix = get_symbol_prefix(stock_code)
    symbol = f"{prefix}{stock_code}"

    if use_cache:
        cached_df = load_kline(stock_code, "weekly")
        if len(cached_df) >= datalen * 0.8:
            return cached_df.iloc[-datalen:].reset_index(drop=True)

    url = (
        f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale=1200&ma=no&datalen={datalen}"
    )

    raw = curl_get(url, extra_headers={"Referer": "https://finance.sina.com.cn"})
    data = json.loads(raw)

    if not data or len(data) < 10:
        return pd.DataFrame()

    rows = []
    for item in data:
        rows.append({
            "日期": item["day"],
            "开盘": float(item["open"]),
            "收盘": float(item["close"]),
            "最高": float(item["high"]),
            "最低": float(item["low"]),
            "成交量": float(item["volume"]),
        })

    df = pd.DataFrame(rows)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    df["成交额"] = df["成交量"] * 100 * (df["开盘"] + df["收盘"]) / 2

    if use_cache:
        cache_kline(stock_code, df, "weekly")

    return df


def fetch_index_data(index_code="000001", datalen=90):
    """获取大盘指数K线数据

    index_code:
    000001 = 上证指数
    399001 = 深证成指
    399006 = 创业板指
    """
    # 指数使用特殊前缀
    if index_code.startswith("000"):
        symbol = f"sh{index_code}"
    elif index_code.startswith("399"):
        symbol = f"sz{index_code}"
    else:
        symbol = f"{get_symbol_prefix(index_code)}{index_code}"

    url = (
        f"https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={datalen}"
    )

    try:
        raw = curl_get(url, extra_headers={"Referer": "https://finance.sina.com.cn"})
        data = json.loads(raw)

        if not data or len(data) < 10:
            return None

        rows = []
        for item in data:
            rows.append({
                "日期": item["day"],
                "开盘": float(item["open"]),
                "收盘": float(item["close"]),
                "最高": float(item["high"]),
                "最低": float(item["low"]),
                "成交量": float(item["volume"]),
            })

        df = pd.DataFrame(rows)
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)

        # 缓存指数数据
        cache_kline(index_code, df, "index_daily")

        return df
    except Exception:
        return None


def fetch_spot_data(stock_code):
    """获取当日实时数据 (腾讯API)，返回 dict
    扩展: 增加更多基本面字段

    关键字段位置:
    1: name, 3: current_price, 4: prev_close, 5: open,
    6: volume(手), 33: high, 34: low, 32: change_pct,
    38: turnover_rate, 39: PE, 49: PB,
    45: total_market_cap(亿)
    """
    prefix = get_symbol_prefix(stock_code)
    symbol = f"{prefix}{stock_code}"

    url = f"https://qt.gtimg.cn/q={symbol}"

    raw = curl_get(url, encoding="gbk", extra_headers={"Referer": "https://gu.qq.com"})

    if not raw or f"v_{symbol}=" not in raw:
        raise ValueError(f"股票 {stock_code} 未返回实时数据")

    start = raw.index('"') + 1
    end = raw.rindex('"')
    content = raw[start:end]
    fields = content.split("~")

    def safe_float(idx, default=0):
        if len(fields) > idx and fields[idx]:
            try:
                return float(fields[idx])
            except ValueError:
                return default
        return default

    def safe_str(idx, default=""):
        if len(fields) > idx and fields[idx]:
            return fields[idx]
        return default

    spot_dict = {
        "代码": stock_code,
        "名称": safe_str(1, stock_code),
        "最新价": safe_float(3),
        "昨收": safe_float(4),
        "今开": safe_float(5),
        "成交量": safe_float(6),  # 手
        "最高": safe_float(33),
        "最低": safe_float(34),
        "涨跌幅": safe_float(32),
        "换手率": safe_float(38),
        "市盈率-动态": safe_float(39),
        "总市值": safe_float(45),  # 亿
        "市净率": safe_float(49),
    }

    # ETF标记: 没有PE/PB概念
    spot_dict["is_etf"] = is_etf(stock_code)

    # 计算涨跌幅 (精确)
    if spot_dict["昨收"] > 0 and spot_dict["最新价"] > 0:
        spot_dict["涨跌幅"] = round((spot_dict["最新价"] - spot_dict["昨收"]) / spot_dict["昨收"] * 100, 2)

    # 成交额
    spot_dict["成交额"] = spot_dict["成交量"] * 100 * (spot_dict["今开"] + spot_dict["最新价"]) / 2

    return spot_dict


def fetch_stock_data(stock_code, period="3m", use_cache=True):
    """完整数据获取：历史K线 + 实时数据 + 大盘指数数据"""
    hist_df = fetch_hist_data(stock_code, period, use_cache=use_cache)
    spot_dict = fetch_spot_data(stock_code)
    stock_name = spot_dict.get("名称", stock_code)

    # 补充换手率
    if "换手率" not in hist_df.columns:
        hist_df["换手率"] = 0
    if "换手率" in spot_dict and spot_dict["换手率"] > 0:
        hist_df["换手率"] = hist_df["换手率"].fillna(spot_dict["换手率"])

    # 缓存基本面数据
    fund_dict = {
        "pe": spot_dict.get("市盈率-动态", 0),
        "pb": spot_dict.get("市净率", 0),
        "turnover_rate": spot_dict.get("换手率", 0),
        "total_market_cap": spot_dict.get("总市值", 0),
    }
    cache_fundamentals(stock_code, fund_dict)

    # 获取大盘指数数据
    index_df = fetch_index_data("000001", datalen=90)

    return hist_df, spot_dict, stock_name, index_df
