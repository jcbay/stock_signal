"""
SQLite 数据库层 - 数据缓存与回测结果存储
零配置嵌入式数据库，与 Python 进程一起运行
"""

import sqlite3
import os
import json
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "stock_data.db")


def get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_conn()
    cursor = conn.cursor()

    # K线数据缓存表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kline_cache (
            stock_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            turnover REAL,
            change_pct REAL,
            amplitude REAL,
            period_type TEXT NOT NULL DEFAULT 'daily',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (stock_code, trade_date, period_type)
        )
    """)

    # 基本面数据缓存表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_cache (
            stock_code TEXT NOT NULL,
            report_date TEXT NOT NULL,
            pe REAL,
            pb REAL,
            roe REAL,
            revenue_growth REAL,
            profit_growth REAL,
            gross_margin REAL,
            debt_ratio REAL,
            operating_cashflow REAL,
            net_profit REAL,
            total_market_cap REAL,
            turnover_rate REAL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (stock_code, report_date)
        )
    """)

    # 回测结果表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT NOT NULL,
            period TEXT NOT NULL,
            model_version TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            total_signals INTEGER,
            win_rate REAL,
            avg_return REAL,
            max_drawdown REAL,
            sharpe_ratio REAL,
            profit_factor REAL,
            results_json TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 自选股表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            stock_code TEXT PRIMARY KEY,
            stock_name TEXT,
            sort_order INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # 评分记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS score_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT NOT NULL,
            stock_name TEXT,
            overall_score REAL,
            recommendation TEXT,
            signal_strength TEXT,
            action_type TEXT,
            trend_veto INTEGER,
            factor_scores_json TEXT,
            market_regime TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def cache_kline(stock_code, df, period_type="daily"):
    """缓存K线数据到数据库"""
    conn = get_conn()
    now = datetime.now().isoformat()

    rows = []
    for _, row in df.iterrows():
        date_str = row["日期"].strftime("%Y-%m-%d") if isinstance(row["日期"], pd.Timestamp) else str(row["日期"])
        rows.append((
            stock_code, date_str,
            row.get("开盘", 0), row.get("收盘", 0),
            row.get("最高", 0), row.get("最低", 0),
            row.get("成交量", 0), row.get("成交额", 0),
            row.get("涨跌幅", 0), row.get("振幅", 0),
            period_type, now
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO kline_cache
        (stock_code, trade_date, open, close, high, low, volume, turnover,
         change_pct, amplitude, period_type, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()


def load_kline(stock_code, period_type="daily", days=None):
    """从缓存加载K线数据"""
    conn = get_conn()

    if days:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = conn.execute("""
            SELECT * FROM kline_cache
            WHERE stock_code = ? AND period_type = ? AND trade_date >= ?
            ORDER BY trade_date
        """, (stock_code, period_type, cutoff)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM kline_cache
            WHERE stock_code = ? AND period_type = ?
            ORDER BY trade_date
        """, (stock_code, period_type)).fetchall()

    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df = df.rename(columns={
        "trade_date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "volume": "成交量",
        "turnover": "成交额", "change_pct": "涨跌幅", "amplitude": "振幅"
    })
    df["日期"] = pd.to_datetime(df["日期"])
    return df


def kline_cache_age(stock_code, period_type="daily"):
    """返回K线缓存最新数据的日期，None表示无缓存"""
    conn = get_conn()
    row = conn.execute("""
        SELECT MAX(trade_date) as latest FROM kline_cache
        WHERE stock_code = ? AND period_type = ?
    """, (stock_code, period_type)).fetchone()
    conn.close()
    return row["latest"] if row and row["latest"] else None


def cache_fundamentals(stock_code, fund_dict, report_date="latest"):
    """缓存基本面数据"""
    conn = get_conn()
    now = datetime.now().isoformat()

    conn.execute("""
        INSERT OR REPLACE INTO fundamentals_cache
        (stock_code, report_date, pe, pb, roe, revenue_growth, profit_growth,
         gross_margin, debt_ratio, operating_cashflow, net_profit,
         total_market_cap, turnover_rate, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stock_code, report_date,
        fund_dict.get("pe"), fund_dict.get("pb"), fund_dict.get("roe"),
        fund_dict.get("revenue_growth"), fund_dict.get("profit_growth"),
        fund_dict.get("gross_margin"), fund_dict.get("debt_ratio"),
        fund_dict.get("operating_cashflow"), fund_dict.get("net_profit"),
        fund_dict.get("total_market_cap"), fund_dict.get("turnover_rate"),
        now
    ))
    conn.commit()
    conn.close()


def load_fundamentals(stock_code):
    """加载最新的基本面缓存数据"""
    conn = get_conn()
    row = conn.execute("""
        SELECT * FROM fundamentals_cache
        WHERE stock_code = ?
        ORDER BY updated_at DESC LIMIT 1
    """, (stock_code,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_backtest_result(result_dict):
    """保存回测结果"""
    conn = get_conn()
    now = datetime.now().isoformat()

    conn.execute("""
        INSERT INTO backtest_results
        (stock_code, period, model_version, start_date, end_date,
         total_signals, win_rate, avg_return, max_drawdown, sharpe_ratio,
         profit_factor, results_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result_dict["stock_code"], result_dict["period"],
        result_dict["model_version"], result_dict["start_date"],
        result_dict["end_date"], result_dict["total_signals"],
        result_dict["win_rate"], result_dict["avg_return"],
        result_dict["max_drawdown"], result_dict["sharpe_ratio"],
        result_dict["profit_factor"],
        json.dumps(result_dict.get("results_json", {}), ensure_ascii=False),
        now
    ))
    conn.commit()
    conn.close()


def load_backtest_result(stock_code, period="3m"):
    """加载最近的回测结果"""
    conn = get_conn()
    row = conn.execute("""
        SELECT * FROM backtest_results
        WHERE stock_code = ? AND period = ?
        ORDER BY created_at DESC LIMIT 1
    """, (stock_code, period)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_score_history(score_dict):
    """保存评分记录"""
    conn = get_conn()
    now = datetime.now().isoformat()

    conn.execute("""
        INSERT INTO score_history
        (stock_code, stock_name, overall_score, recommendation, signal_strength,
         action_type, trend_veto, factor_scores_json, market_regime, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        score_dict["stock_code"], score_dict["stock_name"],
        score_dict["overall_score"], score_dict["recommendation"],
        score_dict["signal_strength"], score_dict["action_type"],
        int(score_dict.get("trend_veto", False)),
        json.dumps(score_dict.get("factor_scores", {}), ensure_ascii=False),
        score_dict.get("market_regime", "unknown"),
        now
    ))
    conn.commit()
    conn.close()


def load_score_history(stock_code, limit=10):
    """加载最近的评分记录"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM score_history
        WHERE stock_code = ?
        ORDER BY created_at DESC LIMIT ?
    """, (stock_code, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# 自选股管理
# ============================================================

def add_watchlist(stock_code, stock_name=None):
    """添加股票到自选股列表"""
    conn = get_conn()
    now = datetime.now().isoformat()

    # 获取当前最大排序值
    row = conn.execute("SELECT MAX(sort_order) as max_order FROM watchlist").fetchone()
    max_order = row["max_order"] if row and row["max_order"] is not None else 0

    conn.execute("""
        INSERT OR REPLACE INTO watchlist (stock_code, stock_name, sort_order, created_at)
        VALUES (?, ?, ?, ?)
    """, (stock_code, stock_name, max_order + 1, now))
    conn.commit()
    conn.close()


def remove_watchlist(stock_code):
    """从自选股列表移除"""
    conn = get_conn()
    conn.execute("DELETE FROM watchlist WHERE stock_code = ?", (stock_code,))
    conn.commit()
    conn.close()


def list_watchlist():
    """获取全部自选股，按排序顺序返回"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT w.*, s.overall_score, s.recommendation, s.signal_strength,
               s.action_type, s.trend_veto, s.created_at as score_time
        FROM watchlist w
        LEFT JOIN (
            SELECT stock_code, overall_score, recommendation, signal_strength,
                   action_type, trend_veto, created_at
            FROM score_history
            WHERE id IN (SELECT MAX(id) FROM score_history GROUP BY stock_code)
        ) s ON w.stock_code = s.stock_code
        ORDER BY w.sort_order
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def reorder_watchlist(stock_code, new_order):
    """调整自选股排序"""
    conn = get_conn()
    conn.execute("UPDATE watchlist SET sort_order = ? WHERE stock_code = ?",
                 (new_order, stock_code))
    conn.commit()
    conn.close()


# 启动时自动初始化
init_db()
