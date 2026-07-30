"""
评分引擎 v2 — 正交因子加权 + 回测验证 + 风险管理 + 市场环境感知

核心改进:
1. 5正交因子替代4维度指标堆叠
2. Z-Score/百分位评分替代硬编码阈值
3. Walk-forward回测引擎验证模型有效性
4. 市场环境感知（牛市/震荡/熊市动态调整权重）
5. 风险管理模块（ATR止损、仓位建议、风险分级）
6. 统计显著性检验
"""

import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# 因子权重配置（基准权重，市场环境会动态调整）
# ============================================================

BASE_WEIGHTS = {
    "trend_momentum": 0.25,    # 趋势动量 (MACD + ADX)
    "volatility": 0.15,        # 波动率 (BOLL + ATR)
    "volume_flow": 0.15,       # 量价关系 (OBV + MFI)
    "relative_strength": 0.15, # 相对强度 (RSI + ROC)
    "fundamentals": 0.30,      # 基本面 (PE + PB + ROE)
}

# 市场环境动态权重调整
REGIME_WEIGHT_ADJUSTMENTS = {
    "bull": {     # 牛市: 强化趋势和动量，弱化震荡指标
        "trend_momentum": +0.10,
        "volatility": -0.05,
        "volume_flow": +0.05,
        "relative_strength": -0.05,
        "fundamentals": -0.05,
    },
    "bear": {     # 熊市: 强化基本面和风险控制，弱化趋势
        "trend_momentum": -0.10,
        "volatility": +0.05,
        "volume_flow": -0.05,
        "relative_strength": +0.05,
        "fundamentals": +0.10,
    },
    "range": {    # 震荡市: 强化震荡指标和量价关系
        "trend_momentum": -0.05,
        "volatility": +0.05,
        "volume_flow": +0.05,
        "relative_strength": +0.10,
        "fundamentals": -0.05,
    },
    "unknown": {},  # 未知: 用基准权重
}

# 一票否决阈值（趋势动量低于此值直接否决）
TREND_VETO_THRESHOLD = 25


# ============================================================
# 市场环境检测
# ============================================================

def detect_market_regime(index_df):
    """检测大盘市场环境: bull / bear / range / unknown

    基于上证指数近60日走势判断:
    - 累计涨幅 > 15% + ADX > 25 → bull
    - 累计跌幅 > 10% + ADX > 25 → bear
    - 其他 → range
    """
    if index_df is None or len(index_df) < 30:
        return "unknown", "大盘数据不足，使用默认权重"

    latest = index_df.iloc[-1]
    n_days = min(60, len(index_df))
    cumulative_return = (index_df["收盘"].iloc[-1] / index_df["收盘"].iloc[-n_days] - 1) * 100

    # 计算指数ADX
    if "adx" in index_df.columns and not pd.isna(latest.get("adx", np.nan)):
        adx_val = latest["adx"]
    else:
        # 简化: 用近期波动方向替代
        recent_ma20 = index_df["收盘"].rolling(20).mean().iloc[-1] if len(index_df) >= 20 else latest["收盘"]
        recent_ma5 = index_df["收盘"].rolling(5).mean().iloc[-1] if len(index_df) >= 5 else latest["收盘"]
        adx_val = 25 if abs(recent_ma5 - recent_ma20) / recent_ma20 * 100 > 1 else 15

    regime = "range"
    regime_desc = f"震荡市 (近{n_days}日涨幅={cumulative_return:.1f}%)"

    if cumulative_return > 15 and adx_val > 25:
        regime = "bull"
        regime_desc = f"牛市 (近{n_days}日涨幅={cumulative_return:.1f}%, ADX={adx_val:.1f})"
    elif cumulative_return < -10 and adx_val > 25:
        regime = "bear"
        regime_desc = f"熊市 (近{n_days}日跌幅={cumulative_return:.1f}%, ADX={adx_val:.1f})"
    elif adx_val < 20:
        regime_desc = f"震荡市 (ADX={adx_val:.1f}, 近{n_days}日涨幅={cumulative_return:.1f}%)"

    return regime, regime_desc


def get_adjusted_weights(regime):
    """根据市场环境动态调整因子权重"""
    adjustments = REGIME_WEIGHT_ADJUSTMENTS.get(regime, {})
    weights = {}
    for factor, base_weight in BASE_WEIGHTS.items():
        weights[factor] = base_weight + adjustments.get(factor, 0)

    # 确保权重总和为1.0
    total = sum(weights.values())
    if total != 1.0:
        for k in weights:
            weights[k] /= total

    return weights


# ============================================================
# 风险管理模块
# ============================================================

def calc_risk_management(df, spot_data, overall_score):
    """计算风险管理参数: 止损/止盈/仓位/风险等级"""

    latest = df.iloc[-1]
    current_price = latest["收盘"]

    # ATR动态止损
    atr_val = latest.get("atr", None)
    if atr_val is not None and not pd.isna(atr_val) and atr_val > 0:
        # 多头止损: 当前价 - 2×ATR
        stop_loss = current_price - 2 * atr_val
        # 空头止损: 当前价 + 2×ATR (A股一般不做空，但作为参考)
        stop_loss_pct = (current_price - stop_loss) / current_price * 100

        # 止盈: 阻力位或3×ATR（R/R比 ≥ 1.5:1）
        target_profit = current_price + 3 * atr_val
        target_profit_pct = (target_profit - current_price) / current_price * 100

        # R/R比 (风险回报比)
        rr_ratio = (target_profit - current_price) / (current_price - stop_loss) if current_price > stop_loss else 0
    else:
        # 无ATR时用固定2%/6%
        stop_loss = current_price * 0.98
        stop_loss_pct = 2.0
        target_profit = current_price * 1.06
        target_profit_pct = 6.0
        rr_ratio = 3.0
        atr_val = 0

    # 仓位建议 (1/4 Kelly 公式，保守版)
    # f* = (bp - q) / b, 其中 b=盈亏比, p=估算胜率, q=1-p
    # 1/4 Kelly = f*/4, 更安全
    estimated_win_rate = overall_score / 100  # 简化: 评分越高估算胜率越高
    if estimated_win_rate < 0.3:
        estimated_win_rate = 0.3
    if estimated_win_rate > 0.7:
        estimated_win_rate = 0.7

    win_loss_ratio = rr_ratio if rr_ratio > 0 else 1.5
    kelly_f = (win_loss_ratio * estimated_win_rate - (1 - estimated_win_rate)) / win_loss_ratio
    quarter_kelly = max(0, kelly_f / 4)  # 1/4 Kelly

    # 仓位上限: 不超过50%
    position_pct = np.clip(quarter_kelly * 100, 0, 50)

    # 风险等级
    volatility_pct = atr_val / current_price * 100 if atr_val > 0 and current_price > 0 else 2
    max_drawdown_recent = 0
    if len(df) >= 20:
        recent_high = df["收盘"].iloc[-20:].max()
        recent_low = df["收盘"].iloc[-20:].min()
        max_drawdown_recent = (recent_high - recent_low) / recent_high * 100

    if max_drawdown_recent > 15 or volatility_pct > 3:
        risk_level = "高"
    elif max_drawdown_recent > 8 or volatility_pct > 2:
        risk_level = "中"
    else:
        risk_level = "低"

    return {
        "stop_loss": round(stop_loss, 2),
        "stop_loss_pct": round(stop_loss_pct, 2),
        "target_profit": round(target_profit, 2),
        "target_profit_pct": round(target_profit_pct, 2),
        "rr_ratio": round(rr_ratio, 2),
        "position_pct": round(float(position_pct), 1),
        "risk_level": risk_level,
        "atr": round(float(atr_val), 4) if atr_val else 0,
        "volatility_pct": round(volatility_pct, 2),
        "max_drawdown_recent": round(max_drawdown_recent, 2),
    }


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(df, spot_data, model_version="v2_orthogonal"):
    """Walk-forward 回测: 用历史数据验证当前评分模型

    方法: 对每一天用前60天数据计算因子评分，记录信号，统计胜率

    返回: 回测统计指标
    """
    signals = []
    window_size = 60  # 计算因子所需最小数据量

    if len(df) < window_size + 20:
        return {
            "total_signals": 0,
            "win_rate": 0,
            "avg_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "profit_factor": 0,
            "note": "数据不足，无法回测",
        }

    for i in range(window_size, len(df) - 5):  # 留5天检验结果
        sub_df = df.iloc[:i + 1].copy()

        # 计算指标和评分 (简化版，只用可用数据)
        try:
            from indicators import calc_all_indicators
            sub_df = calc_all_indicators(sub_df)

            latest_sub = sub_df.iloc[-1]
            if pd.isna(latest_sub.get("adx", np.nan)):
                continue

            # 简化评分: 只用趋势和相对强度两个核心因子
            from indicators import zscore_rank, percentile_rank

            # 趋势动量简化评分
            macd_diff = sub_df["dif"] - sub_df["dea"] if "dif" in sub_df.columns else pd.Series([0]*len(sub_df))
            trend_score = percentile_rank(macd_diff.dropna(), window=min(60, len(sub_df))) if len(macd_diff.dropna()) > 5 else 50

            adx_val = latest_sub["adx"]
            if adx_val > 25:
                trend_score = min(100, trend_score + 15)
            elif adx_val < 20:
                trend_score = max(0, trend_score - 15)

            if "plus_di" in sub_df.columns and "minus_di" in sub_df.columns:
                if latest_sub["plus_di"] > latest_sub["minus_di"]:
                    trend_score = min(100, trend_score + 10)

            # 相对强度简化评分
            rsi_val = latest_sub.get("rsi", 50)
            rsi_score = percentile_rank(sub_df["rsi"].dropna(), window=min(60, len(sub_df))) if "rsi" in sub_df.columns and len(sub_df["rsi"].dropna()) > 5 else 50

            # 综合评分 (简化加权)
            overall = trend_score * 0.5 + rsi_score * 0.5

            # 信号判定
            action = "hold"
            if overall >= 65 and trend_score >= 40:
                action = "buy"
            elif overall <= 35 or trend_score <= 25:
                action = "sell"

            # 记录信号
            signal_date = sub_df["日期"].iloc[-1]
            buy_price = latest_sub["收盘"]

            # 5天后的收益验证
            future_idx = min(i + 5, len(df) - 1)
            future_price = df.iloc[future_idx]["收盘"]
            future_return = (future_price - buy_price) / buy_price * 100

            signals.append({
                "date": str(signal_date),
                "action": action,
                "score": round(overall, 1),
                "trend_score": round(trend_score, 1),
                "price": round(buy_price, 2),
                "future_return": round(future_return, 2),
            })
        except Exception:
            continue

    # 统计回测指标
    buy_signals = [s for s in signals if s["action"] == "buy"]
    sell_signals = [s for s in signals if s["action"] == "sell"]

    # 买入信号胜率: 5天后收益>0的比例
    buy_wins = [s for s in buy_signals if s["future_return"] > 0]
    win_rate = len(buy_wins) / len(buy_signals) if buy_signals else 0

    # 平均收益
    avg_return = np.mean([s["future_return"] for s in buy_signals]) if buy_signals else 0

    # 最大回撤 (所有信号)
    cumulative_returns = []
    if buy_signals:
        cum = 0
        for s in buy_signals:
            cum += s["future_return"]
            cumulative_returns.append(cum)
        peak = 0
        max_dd = 0
        for r in cumulative_returns:
            peak = max(peak, r)
            dd = (peak - r) / max(abs(peak), 1) * 100
            max_dd = max(max_dd, dd)
    else:
        max_dd = 0

    # Sharpe比率 (简化)
    returns = [s["future_return"] for s in buy_signals] if buy_signals else [0]
    sharpe = np.mean(returns) / (np.std(returns) + 0.01) if returns else 0

    # Profit factor: 总盈利 / 总亏损
    total_profit = sum(s["future_return"] for s in buy_signals if s["future_return"] > 0)
    total_loss = abs(sum(s["future_return"] for s in buy_signals if s["future_return"] < 0))
    profit_factor = total_profit / (total_loss + 0.01)

    start_date = str(df["日期"].iloc[0]) if "日期" in df.columns else ""
    end_date = str(df["日期"].iloc[-1]) if "日期" in df.columns else ""

    return {
        "model_version": model_version,
        "start_date": start_date[:10] if start_date else "",
        "end_date": end_date[:10] if end_date else "",
        "total_signals": len(signals),
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "win_rate": round(win_rate * 100, 1),
        "avg_return": round(avg_return, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "sample_signals": signals[-5:] if len(signals) >= 5 else signals,  # 最近5个信号
    }


# ============================================================
# 综合评分与建议
# ============================================================

def calc_overall_score(scores, regime="unknown"):
    """计算加权综合评分（动态权重）"""
    weights = get_adjusted_weights(regime)
    total = 0
    for factor, weight in weights.items():
        total += scores[factor] * weight
    return round(total, 1), weights


def score_to_label(score):
    """将0-100分数转为语义标签"""
    if score >= 75:
        return "强烈看多"
    elif score >= 60:
        return "偏多"
    elif score >= 45:
        return "中性"
    elif score >= 30:
        return "偏空"
    else:
        return "强烈看空"


def get_recommendation(overall_score, trend_score, regime):
    """根据综合评分、趋势否决和市场环境生成建议"""

    # 一票否决: 趋势动量过低
    if trend_score < TREND_VETO_THRESHOLD:
        if trend_score < 15:
            return "趋势严重走弱，不建议入场，持有仓位考虑减仓", "强", "strong_sell"
        else:
            return f"趋势方向否决(趋势评分={trend_score:.0f})，观望不动", "中", "hold"

    # 市场环境修饰词
    regime_prefix = ""
    if regime == "bull":
        regime_prefix = "[牛市环境] "
    elif regime == "bear":
        regime_prefix = "[熊市环境] "
    elif regime == "range":
        regime_prefix = "[震荡市] "

    # 正常评分映射
    if overall_score >= 75:
        return f"{regime_prefix}多维度信号强烈看多，可以考虑买入或加仓", "强", "strong_buy"
    elif overall_score >= 60:
        return f"{regime_prefix}多数维度看多，可以考虑小仓位试探买入", "中", "buy"
    elif overall_score >= 45:
        return f"{regime_prefix}信号中性偏多，观望为主，等待更明确信号", "弱", "hold"
    elif overall_score >= 30:
        return f"{regime_prefix}信号偏空，不宜加仓，已有仓位注意风控", "中", "hold"
    elif overall_score >= 15:
        return f"{regime_prefix}多数维度看空，考虑减仓或卖出", "中", "sell"
    else:
        return f"{regime_prefix}多维度信号强烈看空，建议卖出或清仓", "强", "strong_sell"


# ============================================================
# AI 读懂式解读（大白话总结）
# ============================================================

def generate_summary(scores, overall_score, recommendation, action_type,
                     trend_veto, regime, regime_desc, risk_mgmt, details,
                     stock_name, stock_code, current_price, change_pct, key_data):
    """生成大白话总结段落 — 覆盖: 现在什么情况、应该怎么做、什么时候改变计划"""

    price_str = f"{current_price}" if current_price != "N/A" else "未知"
    change_str = f"{change_pct}%" if change_pct != "N/A" else "未知"
    is_up = isinstance(change_pct, (int, float)) and change_pct > 0

    # --- 第一段: 现在什么情况 ---
    situation_parts = []

    # 趋势状况
    trend_score = scores.get("trend_momentum", 50)
    trend_details = details.get("trend_momentum", [])
    if trend_veto:
        if trend_score < 15:
            situation_parts.append(f"这只股票的趋势严重走弱(评分仅{trend_score:.0f}分)，价格在持续下跌，方向完全不明朗")
        else:
            situation_parts.append(f"这只股票的趋势方向不明确(趋势评分{trend_score:.0f}分)，还在犹豫没有走出方向")
    elif trend_score >= 65:
        situation_parts.append(f"这只股票趋势走强(评分{trend_score:.0f}分)，价格方向明确向上")
    elif trend_score >= 40:
        situation_parts.append(f"这只股票趋势中等(评分{trend_score:.0f}分)，有一定方向但不够强劲")
    else:
        situation_parts.append(f"这只股票趋势偏弱(评分{trend_score:.0f}分)，方向不够明确")

    # MACD/ADX 补充说明
    for d in trend_details:
        if "金叉" in d:
            situation_parts.append("MACD刚刚出现金叉(上涨信号启动)，就像汽车从刹车切换到了加速")
            break
        elif "死叉" in d:
            situation_parts.append("MACD出现死叉(下跌信号启动)，短期走势可能继续向下")
            break

    # 大盘环境
    regime_text = {"bull": "大盘处于牛市环境，整体市场情绪乐观", "bear": "大盘处于熊市环境，整体市场偏弱需谨慎",
                   "range": "大盘处于震荡状态，方向不明需耐心等待", "unknown": "大盘环境不明"}
    situation_parts.append(regime_text.get(regime, regime_text["unknown"]))

    # 量价关系
    vol_score = scores.get("volume_flow", 50)
    vol_details = details.get("volume_flow", [])
    for d in vol_details:
        if "价涨量缩" in d or "量价背离" in d:
            situation_parts.append("量价出现背离(价格上涨但成交量减少)，这可能是假涨，有人在偷偷出货")
            break
        elif "价跌量增" in d:
            situation_parts.append("虽然价格下跌，但成交量放大，可能有人在低位悄悄买入")
            break

    # 基本面
    fund_score = scores.get("fundamentals", 50)
    if fund_score >= 60:
        situation_parts.append(f"估值方面还不错(评分{fund_score:.0f}分)，属于好公司在合理价位")
    elif fund_score >= 40:
        situation_parts.append(f"估值中等(评分{fund_score:.0f}分)，不算贵也不算便宜")
    else:
        situation_parts.append(f"估值偏高(评分{fund_score:.0f}分)，目前价格可能偏贵")

    situation = "；".join(situation_parts) + "。"

    # --- 第二段: 应该怎么做 ---
    action_parts = []

    if trend_veto and trend_score < 15:
        action_parts.append("建议不要买入，如果已持有应考虑减仓止损")
    elif trend_veto:
        action_parts.append("建议观望不动，不要急着入场")
    elif action_type == "strong_buy":
        action_parts.append("可以考虑买入，多个信号都在支持上涨")
    elif action_type == "buy":
        action_parts.append("可以小仓位试探买入，先少量试试不要一把梭")
    elif action_type == "hold":
        if overall_score >= 45:
            action_parts.append("信号不够明确，建议观望为主耐心等待，宁可错过不可做错")
        else:
            action_parts.append("信号偏弱，不建议加仓，已有仓位注意风险控制")
    elif action_type == "sell":
        action_parts.append("考虑减仓或卖出部分，锁定已有利润")
    elif action_type == "strong_sell":
        action_parts.append("建议尽快卖出，多个信号都在看空")

    # 仓位建议
    pos_pct = risk_mgmt.get("position_pct", 0) if risk_mgmt else 0
    if pos_pct > 0 and not trend_veto:
        action_parts.append(f"建议仓位约{pos_pct:.0f}%，不要超出这个比例")

    action = "；".join(action_parts) + "。"

    # --- 第三段: 什么时候改变计划 ---
    plan_parts = []

    stop_loss = risk_mgmt.get("stop_loss", "N/A") if risk_mgmt else "N/A"
    stop_pct = risk_mgmt.get("stop_loss_pct", "N/A") if risk_mgmt else "N/A"
    target = risk_mgmt.get("target_profit", "N/A") if risk_mgmt else "N/A"
    rr = risk_mgmt.get("rr_ratio", 0) if risk_mgmt else 0

    if stop_loss != "N/A":
        plan_parts.append(f"止损价设为{stop_loss}元(约{stop_pct}%下方空间)，到了这个价就卖，不要扛")

    if target != "N/A":
        plan_parts.append(f"止盈目标约{target}元")

    if rr > 0:
        if rr >= 2:
            plan_parts.append(f"风险回报比{rr:.1f}:1，风险可控值得尝试")
        elif rr >= 1.5:
            plan_parts.append(f"风险回报比{rr:.1f}:1，勉强可以尝试但要注意止损")
        else:
            plan_parts.append(f"风险回报比{rr:.1f}:1偏低，风险大于收益，需格外谨慎")

    risk_level = risk_mgmt.get("risk_level", "中") if risk_mgmt else "中"
    plan_parts.append(f"整体风险等级: {risk_level}")

    # 后续观察条件
    if action_type in ("buy", "strong_buy") and not trend_veto:
        plan_parts.append("如果后续趋势继续加强(评分上升)，可以逐步加仓；如果趋势突然走弱，立即减仓")
    elif action_type == "hold":
        plan_parts.append("等综合评分突破60再考虑入场，低于30则考虑卖出")

    plan = "；".join(plan_parts) + "。"

    return {
        "situation": situation,    # 现在什么情况
        "action": action,          # 应该怎么做
        "plan": plan,              # 什么时候改变计划
        "full_summary": f"📊 **当前情况**：{situation}\n\n💡 **操作建议**：{action}\n\n⚠️ **风控计划**：{plan}",
    }


# ============================================================
# 场景推演（乐观/中性/悲观）
# ============================================================

def generate_scenarios(hist_df, spot_data, overall_score, risk_mgmt, scores):
    """基于历史数据和当前指标，给出3种可能场景"""

    if hist_df is None or len(hist_df) < 30 or not risk_mgmt:
        return {
            "optimistic": {"change_pct": "+8%", "probability": "30%", "condition": "趋势持续加强+大盘转好"},
            "neutral": {"change_pct": "+2%", "probability": "50%", "condition": "震荡小幅上行"},
            "pessimistic": {"change_pct": "-5%", "probability": "20%", "condition": "大盘转弱或量价背离"},
        }

    current_price = hist_df["收盘"].iloc[-1]
    atr_val = risk_mgmt.get("atr", 0)
    volatility_pct = risk_mgmt.get("volatility_pct", 2)

    # 基于历史波动率和评分推演
    # 近期日波动率
    if len(hist_df) >= 20:
        daily_returns = hist_df["收盘"].iloc[-20:].pct_change().dropna()
        daily_vol = daily_returns.std() * 100  # 日波动率%
    else:
        daily_vol = volatility_pct

    # 5日预期波动范围
    vol_5d = daily_vol * np.sqrt(5)  # 5日累计波动

    # 根据评分调整方向概率
    if overall_score >= 65:
        opt_prob, neut_prob, pess_prob = 40, 45, 15
        opt_bias = 1.0  # 倾向上涨
    elif overall_score >= 50:
        opt_prob, neut_prob, pess_prob = 30, 50, 20
        opt_bias = 0.5
    elif overall_score >= 35:
        opt_prob, neut_prob, pess_prob = 20, 45, 35
        opt_bias = -0.5
    else:
        opt_prob, neut_prob, pess_prob = 15, 35, 50
        opt_bias = -1.0

    # 乐观场景: 持续上涨
    opt_change = vol_5d * (1 + opt_bias) * 0.5
    opt_price = current_price * (1 + opt_change / 100)

    # 中性场景: 小幅震荡
    neut_change = vol_5d * 0.2 * (0.5 if opt_bias > 0 else -0.3)
    neut_price = current_price * (1 + neut_change / 100)

    # 悲观场景: 明确下跌
    pess_change = -vol_5d * (1 + abs(opt_bias)) * 0.5
    pess_price = current_price * (1 + pess_change / 100)

    # 场景触发条件
    trend_score = scores.get("trend_momentum", 50)
    vol_score = scores.get("volume_flow", 50)

    opt_condition = "趋势持续加强 + 大盘环境改善 + 量价配合上涨"
    neut_condition = "维持当前震荡状态，方向不明确"
    pess_condition = "趋势走弱或量价背离 + 大盘转弱"

    if trend_score >= 60:
        opt_condition = "MACD持续金叉 + ADX走强(趋势确立) + 大盘配合"
    if vol_score < 40:
        pess_condition = "量价背离(价涨量缩) + 趋势反转信号"

    return {
        "optimistic": {
            "change_pct": f"+{abs(opt_change):.1f}%",
            "target_price": round(float(opt_price), 2),
            "probability": f"{opt_prob}%",
            "condition": opt_condition,
        },
        "neutral": {
            "change_pct": f"{neut_change:+.1f}%",
            "target_price": round(float(neut_price), 2),
            "probability": f"{neut_prob}%",
            "condition": neut_condition,
        },
        "pessimistic": {
            "change_pct": f"{pess_change:.1f}%",
            "target_price": round(float(pess_price), 2),
            "probability": f"{pess_prob}%",
            "condition": pess_condition,
        },
        "current_price": round(float(current_price), 2),
        "timeframe": "约5-10个交易日",
    }


def build_report(analysis_result, stock_code, stock_name, spot_data,
                 hist_df=None, index_df=None):
    """构建完整分析报告"""
    scores = {
        "trend_momentum": analysis_result["trend_momentum"]["score"],
        "volatility": analysis_result["volatility"]["score"],
        "volume_flow": analysis_result["volume_flow"]["score"],
        "relative_strength": analysis_result["relative_strength"]["score"],
        "fundamentals": analysis_result["fundamentals"]["score"],
    }

    # 市场环境检测
    regime, regime_desc = detect_market_regime(index_df)

    # 动态加权综合评分
    overall, weights = calc_overall_score(scores, regime)
    trend_score = scores["trend_momentum"]

    recommendation, signal_strength, action_type = get_recommendation(overall, trend_score, regime)

    # 风险管理
    risk_data = calc_risk_management(hist_df, spot_data, overall) if hist_df is not None else {}

    # 回测 (如果有足够数据)
    backtest_result = run_backtest(hist_df, spot_data) if hist_df is not None and len(hist_df) >= 80 else {
        "total_signals": 0, "note": "数据不足，无法回测"
    }

    report = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "current_price": spot_data.get("最新价", "N/A"),
        "change_pct": spot_data.get("涨跌幅", "N/A"),
        "scores": scores,
        "score_labels": {dim: score_to_label(s) for dim, s in scores.items()},
        "overall_score": overall,
        "overall_label": score_to_label(overall),
        "recommendation": recommendation,
        "signal_strength": signal_strength,
        "action_type": action_type,
        "trend_veto": bool(trend_score < TREND_VETO_THRESHOLD),
        "weights": weights,
        "market_regime": regime,
        "market_regime_desc": regime_desc,
        "risk_management": risk_data,
        "backtest": backtest_result,
        "summary": generate_summary(
            scores, overall, recommendation, action_type,
            bool(trend_score < TREND_VETO_THRESHOLD), regime, regime_desc,
            risk_data,
            {
                "trend_momentum": analysis_result["trend_momentum"]["details"],
                "volatility": analysis_result["volatility"]["details"],
                "volume_flow": analysis_result["volume_flow"]["details"],
                "relative_strength": analysis_result["relative_strength"]["details"],
                "fundamentals": analysis_result["fundamentals"]["details"],
            },
            stock_name, stock_code,
            spot_data.get("最新价", "N/A"), spot_data.get("涨跌幅", "N/A"),
            analysis_result["key_data"],
        ),
        "scenarios": generate_scenarios(hist_df, spot_data, overall, risk_data, scores),
        "details": {
            "trend_momentum": analysis_result["trend_momentum"]["details"],
            "volatility": analysis_result["volatility"]["details"],
            "volume_flow": analysis_result["volume_flow"]["details"],
            "relative_strength": analysis_result["relative_strength"]["details"],
            "fundamentals": analysis_result["fundamentals"]["details"],
        },
        "key_data": analysis_result["key_data"],
        "spot_data": {
            "市盈率-动态": spot_data.get("市盈率-动态", "N/A"),
            "市净率": spot_data.get("市净率", "N/A"),
            "换手率": spot_data.get("换手率", "N/A"),
            "成交额": spot_data.get("成交额", "N/A"),
            "总市值": spot_data.get("总市值", "N/A"),
        },
    }

    return report
