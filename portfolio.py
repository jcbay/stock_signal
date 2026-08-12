"""
持仓管理模块
功能:
1. 获取全部持仓的实时行情
2. 计算盈亏(市值/成本/收益率)
3. 智能建议(止损/止盈/加仓/减仓/组合建议)
"""

import numpy as np
from datetime import datetime

from data_fetcher import fetch_spot_data, is_etf
from db import list_holdings, get_holding, add_holding, update_holding, remove_holding


def get_portfolio_overview():
    """获取持仓组合总览 — 含实时行情、盈亏、智能建议"""
    holdings = list_holdings()
    if not holdings:
        return {
            "total_count": 0,
            "total_cost": 0,
            "total_market_value": 0,
            "total_pnl": 0,
            "total_pnl_pct": 0,
            "holdings": [],
            "suggestions": [],
            "market_regime": "unknown",
        }

    positions = []
    total_cost = 0.0
    total_market_value = 0.0
    total_today_pnl = 0.0

    for h in holdings:
        code = h["stock_code"]
        try:
            spot = fetch_spot_data(code)
            current_price = spot.get("最新价", 0)
            change_pct = spot.get("涨跌幅", 0)
            stock_name = spot.get("名称", h.get("stock_name", code))
            is_etf_flag = is_etf(code)

            qty = h["quantity"]
            cost_price = h["cost_price"]
            cost_value = qty * cost_price
            market_value = qty * current_price
            pnl = market_value - cost_value
            pnl_pct = (pnl / cost_value * 100) if cost_value > 0 else 0
            today_pnl = market_value * change_pct / 100

            total_cost += cost_value
            total_market_value += market_value
            total_today_pnl += today_pnl

            # 获取最近一次评分记录
            from db import load_score_history
            history = load_score_history(code, limit=1)
            latest_score = None
            latest_action = "hold"
            latest_recommendation = ""
            if history:
                latest_score = history[0].get("overall_score")
                latest_action = history[0].get("action_type", "hold")
                latest_recommendation = history[0].get("recommendation", "")

            # 生成单个持仓建议
            advice = generate_position_advice(
                pnl_pct, latest_score, latest_action,
                current_price, cost_price, change_pct,
                is_etf_flag, qty, code, stock_name
            )

            positions.append({
                "stock_code": code,
                "stock_name": stock_name,
                "is_etf": is_etf_flag,
                "quantity": qty,
                "cost_price": round(cost_price, 3),
                "current_price": round(current_price, 3),
                "change_pct": round(change_pct, 2),
                "cost_value": round(cost_value, 2),
                "market_value": round(market_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "today_pnl": round(today_pnl, 2),
                "latest_score": latest_score,
                "latest_action": latest_action,
                "latest_recommendation": latest_recommendation,
                "advice": advice,
                "buy_date": h.get("buy_date", ""),
                "notes": h.get("notes", ""),
            })
        except Exception as e:
            # 行情获取失败时仍展示持仓信息
            qty = h["quantity"]
            cost_price = h["cost_price"]
            cost_value = qty * cost_price
            total_cost += cost_value

            positions.append({
                "stock_code": code,
                "stock_name": h.get("stock_name", code),
                "is_etf": is_etf(code),
                "quantity": qty,
                "cost_price": round(cost_price, 3),
                "current_price": 0,
                "change_pct": 0,
                "cost_value": round(cost_value, 2),
                "market_value": 0,
                "pnl": 0,
                "pnl_pct": 0,
                "today_pnl": 0,
                "latest_score": None,
                "latest_action": "hold",
                "latest_recommendation": "",
                "advice": {"type": "neutral", "text": "行情获取失败，请稍后刷新"},
                "buy_date": h.get("buy_date", ""),
                "notes": h.get("notes", ""),
                "error": str(e),
            })

    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    # 今日涨幅 = 今日盈亏 / 昨日总市值(=今日市值-今日盈亏)
    yesterday_value = total_market_value - total_today_pnl
    total_today_pct = (total_today_pnl / yesterday_value * 100) if yesterday_value > 0 else 0

    # 生成组合级建议
    portfolio_suggestions = generate_portfolio_suggestions(
        positions, total_cost, total_market_value, total_pnl, total_pnl_pct
    )

    # 获取大盘环境
    try:
        from data_fetcher import fetch_index_data
        from indicators import calc_all_indicators
        from scorer import detect_market_regime
        index_df = fetch_index_data("000001", datalen=90)
        if index_df is not None and len(index_df) >= 30:
            index_df = calc_all_indicators(index_df)
            regime, regime_desc = detect_market_regime(index_df)
        else:
            regime, regime_desc = "unknown", "大盘数据不足"
    except Exception:
        regime, regime_desc = "unknown", "获取失败"

    return {
        "total_count": len(positions),
        "total_cost": round(total_cost, 2),
        "total_market_value": round(total_market_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "total_today_pnl": round(total_today_pnl, 2),
        "total_today_pct": round(total_today_pct, 2),
        "market_regime": regime,
        "market_regime_desc": regime_desc,
        "holdings": positions,
        "suggestions": portfolio_suggestions,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_position_advice(pnl_pct, latest_score, latest_action,
                             current_price, cost_price, change_pct,
                             is_etf_flag, qty, code, name):
    """为单个持仓生成智能建议

    返回: {"type": "stop_loss|take_profit|action|neutral", "text": "..."}
    """
    advice_parts = []

    # 1. 止损/深套检查
    if pnl_pct <= -8:
        signal_bullish = (latest_action in ("buy", "strong_buy")
                          and latest_score is not None and latest_score >= 60)

        # 1a. 深套 + 信号转好 → 整合建议(补仓摊薄 / 反弹减仓 / 止损换股)，不再简单判止损
        #     避免"深套止损"与"信号看多买入"相互打架，让持仓视角与简报视角一致
        if pnl_pct <= -15 and signal_bullish:
            avg_cost_if_equal = (cost_price + current_price) / 2   # 等量补仓后的摊薄成本(示例)
            rebound_target = current_price * 1.10                   # 反弹减仓观察位: 现价+10%
            return {
                "type": "recovery",
                "text": (
                    f"深套{abs(pnl_pct):.0f}%但信号转好(评分{latest_score:.0f}·看多)，需主动决策勿躺平："
                    f"①认可则小仓摊薄——等量补仓成本可降至约¥{avg_cost_if_equal:.2f}；"
                    f"②看淡则设反弹减仓价(≈¥{rebound_target:.2f}，现价+10%)减一半降暴露；"
                    f"③基本面恶化则止损换股。注：相关基本面因子偏弱，务必小仓，勿重仓抄底"
                )
            }

        # 1b. 普通止损: 亏损且信号明显偏空
        if latest_action in ("sell", "strong_sell") or (latest_score is not None and latest_score < 35):
            return {
                "type": "stop_loss",
                "text": f"亏损{abs(pnl_pct):.1f}%且信号偏空(评分{latest_score or 'N/A'})，建议止损减仓，不要扛单"
            }
        # 1c. 深度套牢但信号未转好(中性/弱) → 提示止损或补仓
        elif pnl_pct <= -15:
            return {
                "type": "stop_loss",
                "text": f"已亏损{abs(pnl_pct):.1f}%，深度套牢。如果基本面没变可考虑补仓摊低成本，否则建议止损"
            }
        # 1d. 轻度亏损 → 风险提示
        else:
            advice_parts.append(f"当前亏损{abs(pnl_pct):.1f}%，注意控制风险")

    # 2. 止盈检查: 盈利超过20%
    if pnl_pct >= 20:
        if latest_action in ("sell", "strong_sell"):
            return {
                "type": "take_profit",
                "text": f"已盈利{pnl_pct:.1f}%且信号转弱(评分{latest_score or 'N/A'})，建议分批止盈锁定利润"
            }
        elif pnl_pct >= 30:
            return {
                "type": "take_profit",
                "text": f"已盈利{pnl_pct:.1f}%，可考虑减仓1/3~1/2锁定部分利润，剩余仓位用移动止损保护"
            }
        else:
            advice_parts.append(f"已盈利{pnl_pct:.1f}%，可考虑部分止盈")

    # 3. 信号驱动的操作建议
    if latest_score is not None:
        if latest_action in ("strong_buy", "buy") and latest_score >= 60 and pnl_pct < 15:
            advice_parts.append(f"信号看多(评分{latest_score:.0f})，趋势未超买，可考虑持有或小仓位加仓")
        elif latest_action == "hold" and 40 <= latest_score < 60:
            advice_parts.append(f"信号中性(评分{latest_score:.0f})，持有观望，等趋势更明确")
        elif latest_action in ("sell", "strong_sell") and latest_score < 35:
            advice_parts.append(f"信号转弱(评分{latest_score:.0f})，注意减仓防范风险")
        elif latest_score >= 70:
            advice_parts.append(f"信号强烈看多(评分{latest_score:.0f})，持有为主")
    else:
        advice_parts.append("尚未分析过该持仓，建议先运行分析获取信号评分")

    # 4. 今日涨跌提醒
    if abs(change_pct) >= 5:
        if change_pct > 0:
            advice_parts.append(f"今日大涨{change_pct:.1f}%，注意是否过热")
        else:
            advice_parts.append(f"今日大跌{change_pct:.1f}%，检查是否触及止损线")

    text = "；".join(advice_parts) + "。" if advice_parts else "继续持有，保持关注。"

    return {"type": "action", "text": text}


def generate_portfolio_suggestions(positions, total_cost, total_market_value,
                                   total_pnl, total_pnl_pct):
    """生成组合级别的智能建议"""
    suggestions = []

    # 1. 集中度分析
    if total_market_value > 0:
        weights = [(p["stock_code"], p["stock_name"], p["market_value"] / total_market_value * 100)
                    for p in positions if p["market_value"] > 0]
        weights.sort(key=lambda x: x[2], reverse=True)

        if weights and weights[0][2] > 40:
            suggestions.append({
                "type": "concentration",
                "level": "warning",
                "text": f"⚠️ 单只持仓 {weights[0][1]} 占比{weights[0][2]:.0f}%，集中度过高，建议适当分散降低风险"
            })

        if len(weights) <= 2:
            suggestions.append({
                "type": "diversification",
                "level": "info",
                "text": f"持仓仅{len(weights)}只，建议增加到5-8只以分散风险(不同行业/风格搭配)"
            })

    # 2. 盈亏状况分析
    profit_count = sum(1 for p in positions if p["pnl"] > 0)
    loss_count = sum(1 for p in positions if p["pnl"] < 0)

    if total_pnl_pct >= 10:
        suggestions.append({
            "type": "overall_pnl",
            "level": "positive",
            "text": f"组合整体盈利{total_pnl_pct:.1f}%，{profit_count}只盈利/{loss_count}只亏损。可考虑逐步止盈部分获利较大的仓位"
        })
    elif total_pnl_pct <= -10:
        suggestions.append({
            "type": "overall_pnl",
            "level": "danger",
            "text": f"组合整体亏损{abs(total_pnl_pct):.1f}%，{loss_count}只亏损。建议检查亏损持仓的基本面是否变化，考虑止损换股"
        })
    else:
        suggestions.append({
            "type": "overall_pnl",
            "level": "neutral",
            "text": f"组合整体{total_pnl_pct:+.1f}%，{profit_count}盈/{loss_count}亏。维持当前仓位，按信号操作"
        })

    # 3. 止损/止盈检查
    stop_loss_list = [p for p in positions if p["advice"]["type"] == "stop_loss"]
    recovery_list = [p for p in positions if p["advice"]["type"] == "recovery"]
    take_profit_list = [p for p in positions if p["advice"]["type"] == "take_profit"]

    # 3a. 深套但信号转好 → 深套待决策(橙)，不再误报为"止损尽快处理"
    if recovery_list:
        names = "、".join([p["stock_name"] for p in recovery_list[:3]])
        suggestions.append({
            "type": "deep_recovery",
            "level": "warning",
            "text": f"🟠 深套待决策: {names} 深套但信号转好，需主动选择补仓摊薄或设反弹减仓价，勿躺平"
        })

    # 3b. 真正触发止损(深套+信号看空) → 红色提醒
    if stop_loss_list:
        names = "、".join([p["stock_name"] for p in stop_loss_list[:3]])
        suggestions.append({
            "type": "stop_loss_alert",
            "level": "danger",
            "text": f"🔴 止损提醒: {names} 深套且信号偏空，建议尽快处理"
        })

    if take_profit_list:
        names = "、".join([p["stock_name"] for p in take_profit_list[:3]])
        suggestions.append({
            "type": "take_profit_alert",
            "level": "warning",
            "text": f"🟡 止盈提醒: {names} 达到止盈条件，考虑分批卖出锁利"
        })

    # 4. 信号分布
    buy_signals = [p for p in positions if p.get("latest_action") in ("buy", "strong_buy")]
    sell_signals = [p for p in positions if p.get("latest_action") in ("sell", "strong_sell")]
    no_analysis = [p for p in positions if p.get("latest_score") is None]

    if buy_signals:
        suggestions.append({
            "type": "signal_buy",
            "level": "info",
            "text": f"🔵 {len(buy_signals)}只持仓信号看多，趋势良好可继续持有"
        })

    if sell_signals:
        suggestions.append({
            "type": "signal_sell",
            "level": "warning",
            "text": f"🔵 {len(sell_signals)}只持仓信号转弱，注意风控"
        })

    if no_analysis:
        suggestions.append({
            "type": "no_analysis",
            "level": "info",
            "text": f"📊 {len(no_analysis)}只持仓尚未分析过，建议前往分析页获取信号评分"
        })

    return suggestions
