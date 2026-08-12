"""
交易日志 + 信号命中率模块
功能:
1. 记录交易决策时自动获取当前信号评分
2. 信号命中率追踪 — 回溯历史推荐股票后续表现
3. 生成命中率报告(胜率/平均收益/按信号类型分类)
"""

import json
import numpy as np
from datetime import datetime, timedelta

from db import (list_trade_journal, add_trade_journal, delete_trade_journal,
                list_all_score_history, load_kline,
                get_holding, add_holding, update_holding, remove_holding)
from data_fetcher import fetch_spot_data, is_etf


def sync_holding_from_trade(stock_code, stock_name, action_type, quantity, price,
                             trade_date=None):
    """根据交易记录同步更新持仓表 — 单向数据流: 交易 → 持仓

    买入:
      - 无持仓 → 新建持仓
      - 有持仓 → 加仓(加权平均成本)
    卖出:
      - 有持仓 → 减仓(数量≤0则清仓删除)
      - 无持仓 → 仅记录交易日志, 不创建空持仓

    Returns:
        dict: {"synced": bool, "action": str}
    """
    existing = get_holding(stock_code)

    if action_type == "buy":
        if existing:
            # 加仓: 加权平均成本
            old_qty = existing["quantity"]
            old_cost = existing["cost_price"]
            new_qty = old_qty + int(quantity)
            new_cost = (old_qty * old_cost + int(quantity) * float(price)) / new_qty
            update_holding(stock_code, quantity=new_qty, cost_price=round(new_cost, 4))
            return {
                "synced": True,
                "action": f"加仓: {old_qty}→{new_qty}股, 成本价{old_cost:.3f}→{new_cost:.3f}",
            }
        else:
            # 新建持仓
            add_holding(stock_code, stock_name, int(quantity), float(price),
                        buy_date=trade_date or "", notes="交易记录自动创建")
            return {
                "synced": True,
                "action": f"新建持仓: {int(quantity)}股@{float(price):.3f}",
            }

    elif action_type == "sell":
        if existing:
            old_qty = existing["quantity"]
            new_qty = old_qty - int(quantity)
            if new_qty <= 0:
                # 清仓删除
                remove_holding(stock_code)
                return {
                    "synced": True,
                    "action": f"清仓: {old_qty}股全部卖出",
                }
            else:
                # 减仓(成本价不变)
                update_holding(stock_code, quantity=new_qty)
                return {
                    "synced": True,
                    "action": f"减仓: {old_qty}→{new_qty}股",
                }
        else:
            # 无持仓, 仅记录交易日志
            return {
                "synced": False,
                "action": "无对应持仓, 仅记录交易日志",
            }

    return {"synced": False, "action": "未知操作类型"}


def log_trade(stock_code, stock_name, action_type, quantity, price,
              reason="", follow_signal=True, trade_date=None):
    """记录一笔交易日志，自动获取当前信号评分并同步更新持仓

    这是创建/更新持仓的唯一入口 — 持仓数据完全由交易记录驱动。

    Args:
        trade_date: 实际交易日期(如 '2025-08-10')，为空则用当前时间
    """
    # 尝试获取最近一次评分记录
    signal_score = None
    signal_label = ""
    signal_action = ""

    from db import load_score_history
    history = load_score_history(stock_code, limit=1)
    if history:
        latest = history[0]
        signal_score = latest.get("overall_score")
        signal_label = latest.get("recommendation", "")
        signal_action = latest.get("action_type", "hold")

    # 1. 记录交易日志
    add_trade_journal(
        stock_code, stock_name, action_type, quantity, price,
        signal_score=signal_score,
        signal_label=signal_label,
        signal_action=signal_action,
        reason=reason,
        follow_signal=follow_signal,
        trade_date=trade_date,
    )

    # 2. 同步更新持仓(单向: 交易 → 持仓)
    holding_result = sync_holding_from_trade(
        stock_code, stock_name, action_type, quantity, price, trade_date
    )

    return {
        "ok": True,
        "stock_code": stock_code,
        "stock_name": stock_name,
        "action_type": action_type,
        "quantity": quantity,
        "price": price,
        "signal_score": signal_score,
        "signal_label": signal_label,
        "signal_action": signal_action,
        "holding_synced": holding_result["synced"],
        "holding_action": holding_result["action"],
    }


def get_trade_journal_summary():
    """获取交易日志汇总 — 总交易次数、买/卖分布、跟信号 vs 逆信号"""
    journals = list_trade_journal(limit=500)

    total = len(journals)
    buy_count = sum(1 for j in journals if j["action_type"] == "buy")
    sell_count = sum(1 for j in journals if j["action_type"] == "sell")

    # 跟信号 vs 逆信号
    follow_count = sum(1 for j in journals if j.get("follow_signal"))
    against_count = total - follow_count

    # 按股票分组
    stock_stats = {}
    for j in journals:
        code = j["stock_code"]
        if code not in stock_stats:
            stock_stats[code] = {
                "stock_code": code,
                "stock_name": j.get("stock_name", code),
                "buy_count": 0,
                "sell_count": 0,
                "buy_amount": 0,
                "sell_amount": 0,
                "last_action": "",
                "last_date": "",
            }
        s = stock_stats[code]
        if j["action_type"] == "buy":
            s["buy_count"] += 1
            s["buy_amount"] += j["quantity"] * j["price"]
        else:
            s["sell_count"] += 1
            s["sell_amount"] += j["quantity"] * j["price"]
        s["last_action"] = j["action_type"]
        s["last_date"] = j.get("trade_date") or j.get("created_at", "")

    stock_list = sorted(stock_stats.values(), key=lambda x: x["last_date"], reverse=True)

    return {
        "total_trades": total,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "follow_signal_count": follow_count,
        "against_signal_count": against_count,
        "follow_rate": round(follow_count / total * 100, 1) if total > 0 else 0,
        "stock_stats": stock_list,
        "journals": journals[:50],
    }


def calc_signal_hit_rate(days_back=30, forward_days=5):
    """计算信号命中率

    逻辑: 回溯days_back天内的评分记录, 对于评分>=60(买入信号)的推荐,
    检查forward_days天后的涨跌情况, 统计胜率

    Returns:
        dict: 命中率报告
    """
    all_scores = list_all_score_history(limit=2000)
    if not all_scores:
        return {
            "total_signals": 0,
            "message": "暂无评分历史数据",
            "buy_hit_rate": 0,
            "sell_hit_rate": 0,
            "details": [],
        }

    now = datetime.now()
    cutoff = now - timedelta(days=days_back)

    # 筛选时间范围内的记录
    valid_records = []
    for r in all_scores:
        try:
            record_time = datetime.fromisoformat(r["created_at"])
            if record_time >= cutoff:
                valid_records.append(r)
        except Exception:
            continue

    if not valid_records:
        return {
            "total_signals": 0,
            "message": f"最近{days_back}天无评分记录",
            "buy_hit_rate": 0,
            "sell_hit_rate": 0,
            "details": [],
        }

    # 按股票+日期分组, 同一天同一只股票只取最新一条
    seen = set()
    unique_records = []
    for r in valid_records:
        try:
            record_date = datetime.fromisoformat(r["created_at"]).strftime("%Y-%m-%d")
        except Exception:
            continue
        key = (r["stock_code"], record_date)
        if key not in seen:
            seen.add(key)
            unique_records.append(r)

    buy_signals = []
    sell_signals = []
    hold_signals = []

    for r in unique_records:
        score = r.get("overall_score", 50)
        action = r.get("action_type", "hold")

        if action in ("strong_buy", "buy") or score >= 60:
            buy_signals.append(r)
        elif action in ("sell", "strong_sell") or score < 35:
            sell_signals.append(r)
        else:
            hold_signals.append(r)

    # 检查买入信号的后续表现
    buy_hits = 0
    buy_misses = 0
    buy_details = []

    for r in buy_signals:
        result = _check_forward_performance(r, forward_days)
        if result is None:
            continue

        if result["return_pct"] > 0:
            buy_hits += 1
        else:
            buy_misses += 1

        buy_details.append({
            "stock_code": r["stock_code"],
            "stock_name": r.get("stock_name", r["stock_code"]),
            "signal_date": result["signal_date"],
            "signal_score": r.get("overall_score"),
            "signal_action": r.get("action_type"),
            "forward_days": forward_days,
            "entry_price": result["entry_price"],
            "exit_price": result["exit_price"],
            "return_pct": round(result["return_pct"], 2),
            "hit": result["return_pct"] > 0,
        })

    # 检查卖出信号的后续表现 (卖出信号正确 = 后续下跌)
    sell_hits = 0
    sell_misses = 0
    sell_details = []

    for r in sell_signals:
        result = _check_forward_performance(r, forward_days)
        if result is None:
            continue

        if result["return_pct"] < 0:
            sell_hits += 1
        else:
            sell_misses += 1

        sell_details.append({
            "stock_code": r["stock_code"],
            "stock_name": r.get("stock_name", r["stock_code"]),
            "signal_date": result["signal_date"],
            "signal_score": r.get("overall_score"),
            "signal_action": r.get("action_type"),
            "forward_days": forward_days,
            "entry_price": result["entry_price"],
            "exit_price": result["exit_price"],
            "return_pct": round(result["return_pct"], 2),
            "hit": result["return_pct"] < 0,
        })

    total_buy = buy_hits + buy_misses
    total_sell = sell_hits + sell_misses

    buy_hit_rate = round(buy_hits / total_buy * 100, 1) if total_buy > 0 else 0
    sell_hit_rate = round(sell_hits / total_sell * 100, 1) if total_sell > 0 else 0

    # 计算买入信号的平均收益
    buy_avg_return = round(np.mean([d["return_pct"] for d in buy_details]), 2) if buy_details else 0
    sell_avg_return = round(np.mean([d["return_pct"] for d in sell_details]), 2) if sell_details else 0

    return {
        "total_signals": len(unique_records),
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "hold_signals": len(hold_signals),
        "buy_evaluated": total_buy,
        "sell_evaluated": total_sell,
        "buy_hit_rate": buy_hit_rate,
        "sell_hit_rate": sell_hit_rate,
        "buy_avg_return": buy_avg_return,
        "sell_avg_return": sell_avg_return,
        "days_back": days_back,
        "forward_days": forward_days,
        "buy_details": buy_details[:20],
        "sell_details": sell_details[:20],
        "message": "数据充足" if total_buy + total_sell >= 5 else "数据较少,建议多分析几只股票后再评估",
    }


def _check_forward_performance(record, forward_days=5):
    """检查某条评分记录后N天的涨跌情况"""
    try:
        signal_date = datetime.fromisoformat(record["created_at"])
        code = record["stock_code"]

        # 从缓存加载K线数据
        df = load_kline(code, days=forward_days + 30)
        if df.empty:
            return None

        # 找到信号日期附近的K线
        df["日期"] = df["日期"].dt.strftime("%Y-%m-%d") if hasattr(df["日期"].dt, "strftime") else df["日期"]
        signal_date_str = signal_date.strftime("%Y-%m-%d")

        # 找信号当天的K线(或最近的)
        mask = df["日期"] >= signal_date_str
        if mask.any():
            entry_idx = df[mask].index[0]
        else:
            # 信号日期在数据之前,取最后forward_days条
            entry_idx = max(0, len(df) - forward_days - 1)

        entry_row = df.loc[entry_idx]
        entry_price = float(entry_row["收盘"])

        # 找forward_days天后的K线
        exit_idx = entry_idx + forward_days
        if exit_idx >= len(df):
            exit_idx = len(df) - 1

        exit_row = df.loc[exit_idx]
        exit_price = float(exit_row["收盘"])

        if entry_price <= 0:
            return None

        return_pct = (exit_price - entry_price) / entry_price * 100

        return {
            "signal_date": signal_date_str,
            "entry_price": round(entry_price, 3),
            "exit_price": round(exit_price, 3),
            "return_pct": return_pct,
        }
    except Exception:
        return None
