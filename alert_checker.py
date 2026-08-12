"""
智能预警模块
功能:
1. 价格预警 — 股价上穿/下穿阈值
2. 信号变化预警 — 评分突破/跌破阈值 或 信号方向翻转
3. 止损止盈预警 — 持仓亏损/盈利达到阈值
4. 批量检查所有活跃预警,返回触发的预警列表
"""

from datetime import datetime
from db import (list_alert_rules, update_alert_status, update_alert_checked,
                list_holdings, load_score_history)
from data_fetcher import fetch_spot_data


def check_all_alerts():
    """检查所有活跃预警规则,返回触发的预警列表"""
    rules = list_alert_rules(status="active")
    if not rules:
        return {"triggered": [], "checked": 0, "message": "没有活跃的预警规则"}

    triggered_list = []
    checked_count = 0

    # 批量获取行情(按股票代码分组)
    spot_cache = {}

    for rule in rules:
        checked_count += 1
        code = rule["stock_code"]
        alert_type = rule["alert_type"]
        threshold = rule.get("threshold")

        # 获取行情
        if code not in spot_cache:
            try:
                spot_cache[code] = fetch_spot_data(code)
            except Exception:
                spot_cache[code] = None

        spot = spot_cache[code]
        if not spot:
            update_alert_checked(rule["id"])
            continue

        current_price = spot.get("最新价", 0)
        change_pct = spot.get("涨跌幅", 0)
        stock_name = spot.get("名称", rule.get("stock_name", code))

        triggered = False
        trigger_msg = ""

        if alert_type == "price_above" and threshold:
            if current_price >= threshold:
                triggered = True
                trigger_msg = f"{stock_name}({code}) 现价{current_price:.3f} 已突破{threshold:.3f}"

        elif alert_type == "price_below" and threshold:
            if current_price <= threshold:
                triggered = True
                trigger_msg = f"{stock_name}({code}) 现价{current_price:.3f} 已跌破{threshold:.3f}"

        elif alert_type == "signal_buy" and threshold:
            # 评分突破阈值
            history = load_score_history(code, limit=1)
            if history:
                score = history[0].get("overall_score", 0)
                if score >= threshold:
                    triggered = True
                    trigger_msg = f"{stock_name}({code}) 信号评分{score:.0f} 已突破{threshold:.0f}分"

        elif alert_type == "signal_sell" and threshold:
            # 评分跌破阈值
            history = load_score_history(code, limit=1)
            if history:
                score = history[0].get("overall_score", 100)
                if score <= threshold:
                    triggered = True
                    trigger_msg = f"{stock_name}({code}) 信号评分{score:.0f} 已跌破{threshold:.0f}分"

        elif alert_type == "stop_loss" and threshold:
            # 持仓亏损达到阈值(百分比)
            holdings = list_holdings()
            for h in holdings:
                if h["stock_code"] == code:
                    cost = h["cost_price"]
                    if cost > 0:
                        pnl_pct = (current_price - cost) / cost * 100
                        if pnl_pct <= -abs(threshold):
                            triggered = True
                            trigger_msg = f"{stock_name}({code}) 亏损{abs(pnl_pct):.1f}%，达到止损线{-threshold:.0f}%"
                    break

        elif alert_type == "take_profit" and threshold:
            # 持仓盈利达到阈值(百分比)
            holdings = list_holdings()
            for h in holdings:
                if h["stock_code"] == code:
                    cost = h["cost_price"]
                    if cost > 0:
                        pnl_pct = (current_price - cost) / cost * 100
                        if pnl_pct >= threshold:
                            triggered = True
                            trigger_msg = f"{stock_name}({code}) 盈利{pnl_pct:.1f}%，达到止盈线{threshold:.0f}%"
                    break

        elif alert_type == "daily_drop" and threshold:
            # 当日跌幅超过阈值
            if change_pct <= -abs(threshold):
                triggered = True
                trigger_msg = f"{stock_name}({code}) 今日大跌{change_pct:.1f}%"

        elif alert_type == "daily_rise" and threshold:
            # 当日涨幅超过阈值
            if change_pct >= threshold:
                triggered = True
                trigger_msg = f"{stock_name}({code}) 今日大涨{change_pct:.1f}%"

        if triggered:
            update_alert_status(rule["id"], "triggered")
            triggered_list.append({
                "rule_id": rule["id"],
                "stock_code": code,
                "stock_name": stock_name,
                "alert_type": alert_type,
                "threshold": threshold,
                "current_price": round(current_price, 3),
                "message": trigger_msg,
                "triggered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        else:
            update_alert_checked(rule["id"])

    return {
        "triggered": triggered_list,
        "checked": checked_count,
        "total_active": len(rules),
    }


def get_alert_type_label(alert_type):
    """获取预警类型的中文标签"""
    labels = {
        "price_above": "价格上穿",
        "price_below": "价格下穿",
        "signal_buy": "信号看多(评分突破)",
        "signal_sell": "信号看空(评分跌破)",
        "stop_loss": "止损预警",
        "take_profit": "止盈预警",
        "daily_drop": "当日大跌",
        "daily_rise": "当日大涨",
    }
    return labels.get(alert_type, alert_type)


def generate_alert_summary():
    """生成预警汇总 — 用于首页速览面板"""
    rules = list_alert_rules()
    active_count = sum(1 for r in rules if r["status"] == "active")
    triggered_count = sum(1 for r in rules if r["status"] == "triggered")

    # 获取触发的预警(最近5条)
    triggered_rules = [r for r in rules if r["status"] == "triggered"][:5]

    return {
        "total_rules": len(rules),
        "active_count": active_count,
        "triggered_count": triggered_count,
        "triggered_recent": triggered_rules,
    }
