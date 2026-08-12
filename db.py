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


def now_str():
    """统一日期时间格式: yyyy-MM-dd HH:mm:SS (无时区, 无微秒)"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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

    # 持仓表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT NOT NULL,
            stock_name TEXT,
            quantity INTEGER NOT NULL DEFAULT 0,
            cost_price REAL NOT NULL DEFAULT 0,
            buy_date TEXT,
            notes TEXT,
            sort_order INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(stock_code)
        )
    """)

    # 交易日志表 — 记录每次买卖决策 + 当时信号
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT NOT NULL,
            stock_name TEXT,
            action_type TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 0,
            price REAL NOT NULL DEFAULT 0,
            signal_score REAL,
            signal_label TEXT,
            signal_action TEXT,
            reason TEXT,
            follow_signal INTEGER DEFAULT 1,
            trade_date TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 兼容已有数据库: 如果缺少 trade_date 列则补上
    try:
        cursor.execute("ALTER TABLE trade_journal ADD COLUMN trade_date TEXT")
    except Exception:
        pass  # 列已存在

    # 已有数据 trade_date 为空时, 用 created_at 填充
    cursor.execute("UPDATE trade_journal SET trade_date = created_at WHERE trade_date IS NULL OR trade_date = ''")

    # 预警规则表 — 价格/信号/止损止盈预警
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT NOT NULL,
            stock_name TEXT,
            alert_type TEXT NOT NULL,
            threshold REAL,
            status TEXT NOT NULL DEFAULT 'active',
            message TEXT,
            created_at TEXT NOT NULL,
            triggered_at TEXT,
            last_checked TEXT
        )
    """)

    # ============================================================
    # 日期格式迁移: 将旧的 ISO 格式(带T和微秒)统一为 yyyy-MM-dd HH:mm:SS
    # ============================================================
    _migrate_date_format(cursor)

    conn.commit()
    conn.close()


def _migrate_date_format(cursor):
    """将所有日期字段从 ISO 格式(2026-08-11T13:23:37.031026)迁移为标准格式(2026-08-11 13:23:37)"""
    # 需要迁移的 表名 → 日期列名列表
    date_columns = {
        "kline_cache": ["updated_at"],
        "fundamentals_cache": ["updated_at"],
        "backtest_results": ["created_at"],
        "watchlist": ["created_at"],
        "score_history": ["created_at"],
        "holdings": ["created_at", "updated_at"],
        "trade_journal": ["created_at", "trade_date"],
        "alert_rules": ["created_at", "triggered_at", "last_checked"],
    }

    for table, cols in date_columns.items():
        for col in cols:
            try:
                # 只更新含有 'T' 的行(ISO格式标志)
                cursor.execute(
                    f"UPDATE {table} SET {col} = substr({col}, 1, 10) || ' ' || substr({col}, 12, 8) "
                    f"WHERE {col} IS NOT NULL AND instr({col}, 'T') > 0"
                )
            except Exception:
                pass  # 表或列可能不存在, 跳过


def cache_kline(stock_code, df, period_type="daily"):
    """缓存K线数据到数据库"""
    conn = get_conn()
    now = now_str()

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
    now = now_str()

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
    now = now_str()

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
    now = now_str()

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
    now = now_str()

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


# ============================================================
# 持仓管理
# ============================================================

def add_holding(stock_code, stock_name, quantity, cost_price, buy_date="", notes=""):
    """添加持仓"""
    conn = get_conn()
    now = now_str()

    row = conn.execute("SELECT MAX(sort_order) as max_order FROM holdings").fetchone()
    max_order = row["max_order"] if row and row["max_order"] is not None else 0

    conn.execute("""
        INSERT OR REPLACE INTO holdings
        (stock_code, stock_name, quantity, cost_price, buy_date, notes, sort_order, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (stock_code, stock_name, int(quantity), float(cost_price),
          buy_date, notes, max_order + 1, now, now))
    conn.commit()
    conn.close()


def update_holding(stock_code, quantity=None, cost_price=None, buy_date=None, notes=None):
    """更新持仓信息"""
    conn = get_conn()
    now = now_str()

    fields = []
    params = []
    if quantity is not None:
        fields.append("quantity = ?")
        params.append(int(quantity))
    if cost_price is not None:
        fields.append("cost_price = ?")
        params.append(float(cost_price))
    if buy_date is not None:
        fields.append("buy_date = ?")
        params.append(buy_date)
    if notes is not None:
        fields.append("notes = ?")
        params.append(notes)
    fields.append("updated_at = ?")
    params.append(now)
    params.append(stock_code)

    conn.execute(f"UPDATE holdings SET {', '.join(fields)} WHERE stock_code = ?", params)
    conn.commit()
    conn.close()


def remove_holding(stock_code):
    """删除持仓"""
    conn = get_conn()
    conn.execute("DELETE FROM holdings WHERE stock_code = ?", (stock_code,))
    conn.commit()
    conn.close()


def list_holdings():
    """获取全部持仓"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM holdings ORDER BY sort_order
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_holding(stock_code):
    """获取单个持仓"""
    conn = get_conn()
    row = conn.execute("SELECT * FROM holdings WHERE stock_code = ?", (stock_code,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ============================================================
# 交易日志管理
# ============================================================

def add_trade_journal(stock_code, stock_name, action_type, quantity, price,
                      signal_score=None, signal_label="", signal_action="",
                      reason="", follow_signal=True, trade_date=None):
    """添加交易日志记录

    Args:
        trade_date: 实际交易日期(YYYY-MM-DD或ISO格式)，为空则用当前时间
    """
    conn = get_conn()
    now = now_str()
    actual_trade_date = trade_date if trade_date else now

    conn.execute("""
        INSERT INTO trade_journal
        (stock_code, stock_name, action_type, quantity, price,
         signal_score, signal_label, signal_action, reason, follow_signal,
         trade_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (stock_code, stock_name, action_type, int(quantity), float(price),
          signal_score, signal_label, signal_action, reason,
          int(follow_signal), actual_trade_date, now))
    conn.commit()
    conn.close()


def list_trade_journal(stock_code=None, limit=100):
    """获取交易日志列表"""
    conn = get_conn()
    if stock_code:
        rows = conn.execute("""
            SELECT * FROM trade_journal
            WHERE stock_code = ?
            ORDER BY COALESCE(trade_date, created_at) DESC LIMIT ?
        """, (stock_code, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM trade_journal
            ORDER BY COALESCE(trade_date, created_at) DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_trade_journal(journal_id):
    """删除交易日志"""
    conn = get_conn()
    conn.execute("DELETE FROM trade_journal WHERE id = ?", (journal_id,))
    conn.commit()
    conn.close()


def list_all_score_history(limit=500):
    """获取全部评分历史(用于信号命中率计算)"""
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM score_history
        ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ============================================================
# 预警规则管理
# ============================================================

def add_alert_rule(stock_code, stock_name, alert_type, threshold=None, message=""):
    """添加预警规则"""
    conn = get_conn()
    now = now_str()

    conn.execute("""
        INSERT INTO alert_rules
        (stock_code, stock_name, alert_type, threshold, status, message, created_at)
        VALUES (?, ?, ?, ?, 'active', ?, ?)
    """, (stock_code, stock_name, alert_type, threshold, message, now))
    conn.commit()
    conn.close()


def list_alert_rules(status=None):
    """获取预警规则列表"""
    conn = get_conn()
    if status:
        rows = conn.execute("""
            SELECT * FROM alert_rules
            WHERE status = ?
            ORDER BY created_at DESC
        """, (status,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT * FROM alert_rules
            ORDER BY created_at DESC
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_alert_status(rule_id, status, triggered_at=None):
    """更新预警规则状态"""
    conn = get_conn()
    now = now_str()
    conn.execute("""
        UPDATE alert_rules
        SET status = ?, triggered_at = ?, last_checked = ?
        WHERE id = ?
    """, (status, triggered_at or now, now, rule_id))
    conn.commit()
    conn.close()


def delete_alert_rule(rule_id):
    """删除预警规则"""
    conn = get_conn()
    conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
    conn.commit()
    conn.close()


def update_alert_checked(rule_id):
    """更新预警最后检查时间"""
    conn = get_conn()
    now = now_str()
    conn.execute("UPDATE alert_rules SET last_checked = ? WHERE id = ?", (now, rule_id))
    conn.commit()
    conn.close()


# 启动时自动初始化
init_db()
