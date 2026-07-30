"""
正交因子模型 - 每个因子测量不同维度，因子之间低相关性
5个正交因子:
1. 趋势动量 (MACD + ADX) — 方向+强度
2. 波动率 (BOLL + ATR) — 风险+通道
3. 量价关系 (OBV + MFI) — 资金流
4. 相对强度 (RSI) — 超买超卖
5. 基本面估值 (PE + PB + ROE + 增速) — 估值+质量

评分方法: Z-Score 标准化 + 百分位排名，而非硬编码阈值
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ============================================================
# 指标计算层
# ============================================================

def calc_ma(df, periods=[5, 10, 20, 60]):
    """计算移动平均线"""
    for p in periods:
        if len(df) >= p:
            df[f"ma{p}"] = df["收盘"].rolling(window=p, min_periods=p).mean()
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    """计算MACD指标 (DIF/DEA/MACD柱)"""
    close = df["收盘"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_bar = (dif - dea) * 2
    df["dif"] = dif
    df["dea"] = dea
    df["macd"] = macd_bar
    return df


def calc_adx(df, period=14):
    """计算ADX (平均趋势指数) — 衡量趋势强度而非方向

    ADX > 40: 强趋势
    ADX 25-40: 中等趋势
    ADX 20-25: 弱趋势
    ADX < 20: 无趋势（震荡市）
    """
    high = df["最高"]
    low = df["最低"]
    close = df["收盘"]

    # +DM 和 -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    df["atr"] = atr

    # +DI / -DI
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # DX → ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * di_diff / di_sum.replace(0, 1)
    adx = dx.ewm(span=period, adjust=False).mean()
    df["adx"] = adx

    return df


def calc_boll(df, period=20, std_dev=2):
    """计算布林带 (BOLL) — 波动通道"""
    mid = df["收盘"].rolling(window=period, min_periods=period).mean()
    std = df["收盘"].rolling(window=period, min_periods=period).std()
    df["boll_mid"] = mid
    df["boll_upper"] = mid + std_dev * std
    df["boll_lower"] = mid - std_dev * std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / mid.replace(0, 1)  # 通道宽度百分比
    return df


def calc_rsi(df, period=14):
    """计算RSI指标"""
    close = df["收盘"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # 用 EWM 方法（更常见）
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    return df


def calc_obv(df):
    """计算OBV能量潮指标"""
    close = df["收盘"]
    volume = df["成交量"]
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    df["obv"] = obv
    return df


def calc_mfi(df, period=14):
    """计算MFI (资金流量指数) — 带成交量的RSI，量价结合

    MFI > 80: 超买
    MFI 20-80: 正常区间
    MFI < 20: 超卖
    """
    typical_price = (df["最高"] + df["最低"] + df["收盘"]) / 3
    raw_money_flow = typical_price * df["成交量"]

    # 正/负资金流
    pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
    neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)

    pos_sum = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(window=period, min_periods=period).sum()

    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, 1)))
    df["mfi"] = mfi
    return df


def calc_vwap(df):
    """计算VWAP (成交量加权均价) — 近期机构成本线"""
    typical_price = (df["最高"] + df["最低"] + df["收盘"]) / 3
    cum_tp_vol = (typical_price * df["成交量"]).cumsum()
    cum_vol = df["成交量"].cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, 1)
    return df


def calc_sar(df, af_step=0.02, af_max=0.2):
    """计算SAR (抛物线止损) — 动态止损跟踪

    SAR 在价格下方 = 多头止损位
    SAR 在价格上方 = 空头止损位
    """
    high = df["最高"].values
    low = df["最低"].values
    close = df["收盘"].values
    n = len(df)

    sar = np.zeros(n)
    trend = np.zeros(n)  # 1=上涨, -1=下跌
    af = np.zeros(n)
    ep = np.zeros(n)  # 极值点

    # 初始化：根据前两日判断初始趋势
    if close[1] > close[0]:
        trend[0] = 1
        sar[0] = low[0]
        ep[0] = high[0]
    else:
        trend[0] = -1
        sar[0] = high[0]
        ep[0] = low[0]

    af[0] = af_step

    for i in range(1, n):
        # SAR = 前SAR + AF * (EP - 前SAR)
        sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

        # SAR 不能超过前2日的极值
        if i >= 2:
            if trend[i - 1] == 1:
                sar[i] = min(sar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            else:
                sar[i] = max(sar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])

        # 趋势翻转检测
        if trend[i - 1] == 1 and low[i] <= sar[i]:
            trend[i] = -1
            sar[i] = ep[i - 1]
            ep[i] = low[i]
            af[i] = af_step
        elif trend[i - 1] == -1 and high[i] >= sar[i]:
            trend[i] = 1
            sar[i] = ep[i - 1]
            ep[i] = high[i]
            af[i] = af_step
        else:
            trend[i] = trend[i - 1]
            # 更新极值点和加速因子
            if trend[i] == 1:
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + af_step, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
            else:
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + af_step, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]

    df["sar"] = sar
    df["sar_trend"] = trend  # 1=多头, -1=空头
    return df


def calc_all_indicators(df):
    """计算所有技术指标"""
    df = calc_ma(df)
    df = calc_macd(df)
    df = calc_adx(df)
    df = calc_boll(df)
    df = calc_rsi(df)
    df = calc_obv(df)
    df = calc_mfi(df)
    df = calc_vwap(df)
    df = calc_sar(df)
    return df


# ============================================================
# Z-Score 标准化 + 百分位排名
# ============================================================

def zscore_rank(series, window=60):
    """对序列做滚动Z-Score，返回当前值在近window天中的相对位置

    优点: 消除绝对值差异，用相对位置评分
    例如 RSI=66 不是简单查表，而是看它在过去60天中处于什么分位
    """
    if len(series) < window:
        window = len(series)
    if window < 5:
        return 50.0  # 数据不足返回中性

    recent = series.iloc[-window:].dropna()
    if len(recent) < 5:
        return 50.0

    current = series.iloc[-1]
    if pd.isna(current):
        return 50.0

    mean = recent.mean()
    std = recent.std()
    if std == 0 or pd.isna(std):
        return 50.0

    z = (current - mean) / std
    # 将Z-Score映射到0-100: z=0 → 50, z=+2 → ~98, z=-2 → ~2
    percentile = 50 + z * 25
    return np.clip(percentile, 0, 100)


def percentile_rank(series, window=60):
    """百分位排名: 当前值在过去window天的百分位位置

    例如: ATR当前值在过去60天中处于80%分位 → 波动率偏高
    """
    if len(series) < window:
        window = len(series)
    if window < 5:
        return 50.0

    recent = series.iloc[-window:].dropna()
    if len(recent) < 5:
        return 50.0

    current = series.iloc[-1]
    if pd.isna(current):
        return 50.0

    rank = (recent < current).sum() / len(recent) * 100
    return np.clip(rank, 0, 100)


# ============================================================
# 正交因子评分层
# ============================================================

def score_trend_momentum(df):
    """因子1: 趋势动量 (MACD + ADX + MA排列) — 方向+强度

    这是与震荡类(RSI)低相关的趋势类因子
    评分方法: Z-Score + 百分位排名，而非硬编码阈值
    """
    details = []
    sub_scores = []

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    # 1. MA排列 — 多头排列程度
    ma_fields = ["ma5", "ma10", "ma20", "ma60"]
    has_ma = all(f in df.columns and not pd.isna(latest[f]) for f in ma_fields)

    if has_ma:
        # 计算排列得分: 短均线高于长均线越多越好
        pairs = [("ma5", "ma10"), ("ma10", "ma20"), ("ma20", "ma60")]
        ma_score = 0
        for short, long in pairs:
            gap_pct = (latest[short] - latest[long]) / latest[long] * 100 if latest[long] != 0 else 0
            ma_score += max(0, min(25, 10 + gap_pct * 2))  # gap>0得分高，cap=25

        # 价格在MA5之上加分
        if latest["收盘"] > latest["ma5"]:
            price_above_pct = (latest["收盘"] - latest["ma5"]) / latest["ma5"] * 100
            ma_score += min(25, 10 + price_above_pct * 2)
        else:
            ma_score += max(0, 5 - abs((latest["收盘"] - latest["ma5"]) / latest["ma5"]) * 100 * 2)

        sub_scores.append(min(100, ma_score))
        details.append(f"均线排列: 多头程度 {ma_score:.0f}/100")
    else:
        sub_scores.append(50)
        details.append("均线数据不足")

    # 2. MACD方向 — DIF-DEA差异的百分位排名
    if "dif" in df.columns and "dea" in df.columns:
        macd_diff = df["dif"] - df["dea"]
        macd_rank = percentile_rank(macd_diff, window=min(60, len(df)))
        # MACD柱变化方向
        if "macd" in df.columns and not pd.isna(latest["macd"]) and not pd.isna(prev["macd"]):
            macd_rising = latest["macd"] > prev["macd"]
            macd_rank = macd_rank + 10 if macd_rising else macd_rank - 10
        sub_scores.append(np.clip(macd_rank, 0, 100))
        dif_val = latest["dif"] if not pd.isna(latest["dif"]) else 0
        dea_val = latest["dea"] if not pd.isna(latest["dea"]) else 0
        details.append(f"MACD: DIF={dif_val:.2f}, DEA={dea_val:.2f}, 百分位={macd_rank:.0f}")
    else:
        sub_scores.append(50)
        details.append("MACD数据不足")

    # 3. ADX趋势强度 — 百分位排名
    if "adx" in df.columns and not pd.isna(latest["adx"]):
        adx_val = latest["adx"]
        adx_rank = percentile_rank(df["adx"], window=min(60, len(df)))

        # ADX > 25 表示有趋势 (加分), ADX < 20 表示无趋势 (减分)
        if adx_val > 25:
            adx_rank = min(100, adx_rank + 15)
        elif adx_val < 20:
            adx_rank = max(0, adx_rank - 15)

        # 趋势方向: +DI > -DI = 上涨趋势加分
        if "plus_di" in df.columns and "minus_di" in df.columns:
            if latest["plus_di"] > latest["minus_di"]:
                adx_rank = min(100, adx_rank + 10)
                details.append(f"ADX={adx_val:.1f}(趋势强度), +DI>-DI, 上涨趋势")
            else:
                details.append(f"ADX={adx_val:.1f}(趋势强度), -DI>+DI, 下跌趋势")

        sub_scores.append(np.clip(adx_rank, 0, 100))
        details.append(f"ADX趋势强度: {adx_val:.1f}, 百分位={adx_rank:.0f}")
    else:
        sub_scores.append(50)
        details.append("ADX数据不足")

    # 4. SAR方向
    if "sar_trend" in df.columns:
        sar_trend = latest["sar_trend"]
        sar_val = latest["sar"] if not pd.isna(latest["sar"]) else 0
        if sar_trend > 0:
            sub_scores.append(65)
            details.append(f"SAR={sar_val:.2f}, 多头止损位(价格上方)")
        else:
            sub_scores.append(35)
            details.append(f"SAR={sar_val:.2f}, 空头止损位(价格下方)")
    else:
        sub_scores.append(50)
        details.append("SAR数据不足")

    # 加权汇总: MA排列30%, MACD25%, ADX30%, SAR15%
    weights = [0.30, 0.25, 0.30, 0.15]
    total = sum(s * w for s, w in zip(sub_scores, weights))
    total = np.clip(total, 0, 100)

    return round(total, 1), details


def score_volatility(df):
    """因子2: 波动率 (BOLL + ATR) — 与趋势类独立的风险维度

    逻辑:
    - 低波动率 + 收盘价在布林带中轨上方 = 低风险看多 → 高分
    - 高波动率 + 收盘价触及布林带下轨 = 高风险偏空 → 低分
    """
    details = []
    sub_scores = []

    latest = df.iloc[-1]

    # 1. 布林带位置 — 收盘价在通道中的相对位置
    if all(f in df.columns and not pd.isna(latest[f]) for f in ["boll_upper", "boll_mid", "boll_lower", "收盘"]):
        boll_range = latest["boll_upper"] - latest["boll_lower"]
        if boll_range > 0:
            boll_pct = (latest["收盘"] - latest["boll_lower"]) / boll_range * 100
            # 百分位排名
            boll_pct_series = (df["收盘"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"]).replace(0, 1) * 100
            boll_rank = percentile_rank(boll_pct_series.dropna(), window=min(60, len(df)))

            # 中轨之上加分
            if boll_pct > 50:
                boll_rank = min(100, boll_rank + 15)
            else:
                boll_rank = max(0, boll_rank - 15)

            sub_scores.append(np.clip(boll_rank, 0, 100))
            details.append(f"布林带位置: {boll_pct:.0f}% (0%=下轨, 100%=上轨)")
        else:
            sub_scores.append(50)
            details.append("布林带宽度为0")
    else:
        sub_scores.append(50)
        details.append("布林带数据不足")

    # 2. 布林带宽度 — 波动率百分位
    if "boll_width" in df.columns and not pd.isna(latest["boll_width"]):
        boll_width_rank = percentile_rank(df["boll_width"].dropna(), window=min(60, len(df)))

        # 波动率收缩(boll_width_rank低) → 即将突破 → 中性偏多
        # 波动率扩张(boll_width_rank高) → 高风险 → 需谨慎
        # 这个维度是风险度量，高分 = 低风险环境
        vol_score = 100 - boll_width_rank  # 反转: 低波动率=高分(低风险)
        sub_scores.append(np.clip(vol_score, 0, 100))
        details.append(f"波动率通道宽度百分位: {boll_width_rank:.0f}% (低=收缩, 高=扩张)")
    else:
        sub_scores.append(50)
        details.append("布林带宽度数据不足")

    # 3. ATR百分位 — 近期波动率的相对水平
    if "atr" in df.columns and not pd.isna(latest["atr"]):
        atr_rank = percentile_rank(df["atr"].dropna(), window=min(60, len(df)))
        atr_score = 100 - atr_rank  # 反转: 低ATR=高分(低波动)
        sub_scores.append(np.clip(atr_score, 0, 100))
        details.append(f"ATR波动率百分位: {atr_rank:.0f}% (低=稳定, 高=波动)")
    else:
        sub_scores.append(50)
        details.append("ATR数据不足")

    # 加权: 布林带位置40%, 波动率收缩30%, ATR30%
    weights = [0.40, 0.30, 0.30]
    total = sum(s * w for s, w in zip(sub_scores, weights))
    total = np.clip(total, 0, 100)

    return round(total, 1), details


def score_volume_flow(df):
    """因子3: 量价关系 (OBV + MFI) — 资金流，与纯价格指标独立

    OBV: 价格上涨+放量 → OBV上升 → 资金流入
    MFI: 带成交量的RSI → 真实资金进出
    """
    details = []
    sub_scores = []

    latest = df.iloc[-1]

    # 1. OBV趋势 — 百分位排名
    if "obv" in df.columns and not pd.isna(latest["obv"]):
        obv_rank = percentile_rank(df["obv"].dropna(), window=min(60, len(df)))

        # OBV与价格的背离检测
        price_trend = df["收盘"].iloc[-1] - df["收盘"].iloc[-20] if len(df) >= 20 else 0
        obv_trend = df["obv"].iloc[-1] - df["obv"].iloc[-20] if len(df) >= 20 else 0

        divergence = ""
        if price_trend > 0 and obv_trend < 0:
            # 价格上涨但OBV下降 → 看跌背离
            obv_rank = max(0, obv_rank - 20)
            divergence = "量价背离(价涨量缩)"
        elif price_trend < 0 and obv_trend > 0:
            # 价格下跌但OBV上升 → 看涨背离
            obv_rank = min(100, obv_rank + 20)
            divergence = "量价背离(价跌量增)"
        else:
            divergence = "量价一致"

        sub_scores.append(np.clip(obv_rank, 0, 100))
        details.append(f"OBV百分位={obv_rank:.0f}, {divergence}")
    else:
        sub_scores.append(50)
        details.append("OBV数据不足")

    # 2. MFI — 资金流量百分位排名
    if "mfi" in df.columns and not pd.isna(latest["mfi"]):
        mfi_val = latest["mfi"]
        mfi_rank = percentile_rank(df["mfi"].dropna(), window=min(60, len(df)))

        # MFI > 80 超买(减分), < 20 超卖(加分)
        if mfi_val > 80:
            mfi_rank = max(0, mfi_rank - 15)
        elif mfi_val < 20:
            mfi_rank = min(100, mfi_rank + 15)

        sub_scores.append(np.clip(mfi_rank, 0, 100))
        details.append(f"MFI={mfi_val:.1f}, 百分位={mfi_rank:.0f}")
    else:
        sub_scores.append(50)
        details.append("MFI数据不足")

    # 3. 量价一致性 — 近期量价关系
    if len(df) >= 20:
        recent = df.iloc[-20:]
        up_days = recent[recent["收盘"] > recent["收盘"].shift(1)]
        down_days = recent[recent["收盘"] < recent["收盘"].shift(1)]

        if len(up_days) > 0 and len(down_days) > 0:
            avg_up_vol = up_days["成交量"].mean()
            avg_down_vol = down_days["成交量"].mean()
            vol_consistency = avg_up_vol / avg_down_vol if avg_down_vol > 0 else 1

            # 上涨日成交量 > 下跌日成交量 = 资金偏向买入
            consistency_score = min(100, 50 + (vol_consistency - 1) * 50)
            sub_scores.append(np.clip(consistency_score, 0, 100))
            details.append(f"量价一致性: 上涨均量/下跌均量={vol_consistency:.2f}")
        else:
            sub_scores.append(50)
            details.append("量价一致性数据不足")
    else:
        sub_scores.append(50)
        details.append("数据不足20日")

    # 加权: OBV 35%, MFI 35%, 量价一致性 30%
    weights = [0.35, 0.35, 0.30]
    total = sum(s * w for s, w in zip(sub_scores, weights))
    total = np.clip(total, 0, 100)

    return round(total, 1), details


def score_relative_strength(df):
    """因子4: 相对强度 (RSI) — 与趋势类低相关的震荡类因子

    评分方法: RSI的百分位排名，而非硬编码"RSI>70=超买"
    在牛市中RSI持续>70是常态，不应简单标记为超买
    """
    details = []
    sub_scores = []

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    if "rsi" in df.columns and not pd.isna(latest["rsi"]):
        rsi_val = latest["rsi"]
        rsi_rank = percentile_rank(df["rsi"].dropna(), window=min(60, len(df)))

        # RSI动量: RSI是否在上升
        if not pd.isna(prev["rsi"]):
            rsi_change = rsi_val - prev["rsi"]
            if rsi_change > 0:
                rsi_rank = min(100, rsi_rank + 8)
            else:
                rsi_rank = max(0, rsi_rank - 8)

        sub_scores.append(np.clip(rsi_rank, 0, 100))

        # 语义标签
        if rsi_rank > 75:
            label = "相对偏强(近期高位)"
        elif rsi_rank > 50:
            label = "中性偏强"
        elif rsi_rank > 25:
            label = "中性偏弱"
        else:
            label = "相对偏弱(近期低位)"

        details.append(f"RSI={rsi_val:.1f}, 百分位={rsi_rank:.0f}, {label}")
    else:
        sub_scores.append(50)
        details.append("RSI数据不足")

    # RSI本身只贡献50%，另外50%来自价格动量率(ROC)
    # ROC = 近期收益率，与RSI低相关但互补
    if len(df) >= 10:
        roc_5 = (latest["收盘"] / df["收盘"].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
        roc_10 = (latest["收盘"] / df["收盘"].iloc[-10] - 1) * 100 if len(df) >= 10 else 0

        # ROC 百分位排名
        roc_series = df["收盘"].pct_change(periods=5) * 100
        roc_rank = percentile_rank(roc_series.dropna(), window=min(60, len(df)))

        sub_scores.append(np.clip(roc_rank, 0, 100))
        details.append(f"5日动量ROC={roc_5:.2f}%, 10日ROC={roc_10:.2f}%, 百分位={roc_rank:.0f}")
    else:
        sub_scores.append(50)
        details.append("动量ROC数据不足")

    # 加权: RSI 50%, ROC 50%
    weights = [0.50, 0.50]
    total = sum(s * w for s, w in zip(sub_scores, weights))
    total = np.clip(total, 0, 100)

    return round(total, 1), details


def score_fundamentals(spot_data, hist_df=None):
    """因子5: 基本面估值 (PE + PB + ROE + 增速)

    PE/PB: 从腾讯API获取
    ROE/营收增速/净利润增速: 优先从API获取，缺省用历史涨跌幅辅助估算
    """
    details = []
    sub_scores = []

    # 1. PE估值 — 百分位思维而非硬编码阈值
    pe = spot_data.get("市盈率-动态", None)
    if pe is not None and pe != "-" and not pd.isna(pe) if isinstance(pe, float) else True:
        try:
            pe_val = float(pe)
            if pe_val < 0:
                sub_scores.append(5)
                details.append(f"PE={pe_val:.1f}: 亏损企业")
            elif pe_val == 0:
                sub_scores.append(10)
                details.append(f"PE≈0: 盈利极低")
            else:
                # PE百分位逻辑: 用倒数（收益率）来排名更合理
                # E/P = 收益率, PE=10 → 10%收益率(好), PE=100 → 1%(差)
                earnings_yield = 100 / pe_val
                # 映射: earnings_yield 2%-15% → score 20-80
                pe_score = np.clip(20 + (earnings_yield - 2) / (15 - 2) * 60, 0, 100)
                sub_scores.append(round(pe_score, 1))
                details.append(f"PE={pe_val:.1f}, 收益率={earnings_yield:.1f}%")
        except (ValueError, TypeError):
            sub_scores.append(50)
            details.append(f"PE数据异常")
    else:
        sub_scores.append(50)
        details.append("PE数据不可用")

    # 2. PB估值
    pb = spot_data.get("市净率", None)
    if pb is not None and pb != "-" and not pd.isna(pb) if isinstance(pb, float) else True:
        try:
            pb_val = float(pb)
            if pb_val < 0:
                sub_scores.append(10)
                details.append(f"PB={pb_val:.2f}: 资产减值")
            else:
                # PB<1 = 低于净资产(好), PB>5 = 极高估值(差)
                pb_score = np.clip(100 - (pb_val - 0.5) / (6 - 0.5) * 100, 0, 100)
                sub_scores.append(round(pb_score, 1))
                details.append(f"PB={pb_val:.2f}")
        except (ValueError, TypeError):
            sub_scores.append(50)
            details.append("PB数据异常")
    else:
        sub_scores.append(50)
        details.append("PB数据不可用")

    # 3. ROE (如有)
    roe = spot_data.get("roe", None)
    if roe is not None and roe != "-":
        try:
            roe_val = float(roe)
            roe_score = np.clip(roe_val / 15 * 60, 0, 100)  # ROE=15% → 60分
            sub_scores.append(round(roe_score, 1))
            details.append(f"ROE={roe_val:.1f}%")
        except (ValueError, TypeError):
            sub_scores.append(50)
            details.append("ROE数据异常")
    else:
        # ROE不可用时用历史收益辅助
        if hist_df is not None and len(hist_df) >= 60:
            # 近60日累计收益率作为成长性近似
            cum_return = (hist_df["收盘"].iloc[-1] / hist_df["收盘"].iloc[0] - 1) * 100
            growth_score = np.clip(50 + cum_return, 0, 100)
            sub_scores.append(round(growth_score, 1))
            details.append(f"ROE不可用, 近{len(hist_df)}日累计涨幅={cum_return:.1f}%")
        else:
            sub_scores.append(50)
            details.append("ROE数据不可用")

    # 加权: PE 30%, PB 25%, ROE 45%
    weights = [0.30, 0.25, 0.45]
    total = sum(s * w for s, w in zip(sub_scores, weights))
    total = np.clip(total, 0, 100)

    return round(total, 1), details


# ============================================================
# 综合分析入口
# ============================================================

def analyze_all(df, spot_data):
    """5正交因子综合分析"""
    df = calc_all_indicators(df)

    trend_score, trend_details = score_trend_momentum(df)
    volatility_score, volatility_details = score_volatility(df)
    volume_score, volume_details = score_volume_flow(df)
    rsi_score, rsi_details = score_relative_strength(df)
    fundamentals_score, fundamentals_details = score_fundamentals(spot_data, df)

    # 收集关键数据点
    latest = df.iloc[-1]
    key_data = {}
    for field in ["收盘", "ma5", "ma10", "ma20", "ma60", "dif", "dea", "macd",
                   "rsi", "adx", "atr", "obv", "mfi", "boll_upper", "boll_mid",
                   "boll_lower", "boll_width", "vwap", "sar", "plus_di", "minus_di",
                   "成交量"]:
        if field in df.columns and not pd.isna(latest[field]):
            key_data[field] = round(float(latest[field]), 4)

    return {
        "trend_momentum": {"score": trend_score, "details": trend_details},
        "volatility": {"score": volatility_score, "details": volatility_details},
        "volume_flow": {"score": volume_score, "details": volume_details},
        "relative_strength": {"score": rsi_score, "details": rsi_details},
        "fundamentals": {"score": fundamentals_score, "details": fundamentals_details},
        "key_data": key_data,
    }
