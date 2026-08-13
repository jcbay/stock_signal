"""每日 / 盘间简报数据生成。

被两处复用：
- app.py 的 /api/daily_scan 路由（对外提供 JSON）；
- feishu_push.py 的 push_briefing_now（内建飞书推送）。

把原先散落在 daily_scan 里的大段逻辑独立成 build_briefing()，
保证「接口返回」与「推送内容」永远来自同一份数据，避免分叉。
"""
import numpy as np
from datetime import datetime

from data_fetcher import fetch_stock_data, fetch_index_data, is_etf
from indicators import analyze_all, analyze_etf, calc_all_indicators
from scorer import build_report, build_etf_report, detect_market_regime
from db import save_score_history, list_watchlist


def build_briefing():
    """扫描全部自选股，生成简报数据结构（dict）。

    返回与 /api/daily_scan 完全一致的 briefing JSON；
    当自选股列表为空时返回 None（调用方据此提示用户）。
    """
    items = list_watchlist()
    if not items:
        return None

    results = []
    for item in items:
        code = item["stock_code"]
        try:
            hist_df, spot_dict, stock_name, index_df = fetch_stock_data(code, "3m")
            if hist_df.empty or len(hist_df) < 30:
                results.append({
                    "stock_code": code, "stock_name": item.get("stock_name", code),
                    "error": "数据不足", "overall_score": None,
                    "recommendation": "无法分析", "action_type": "hold",
                    "signal_strength": "无"
                })
                continue

            if index_df is not None and len(index_df) >= 30:
                index_df = calc_all_indicators(index_df)

            # ETF / 个股路由
            etf_flag = is_etf(code)
            if etf_flag:
                analysis_result = analyze_etf(hist_df, spot_dict)
                report = build_etf_report(analysis_result, code, item.get("stock_name", stock_name),
                                          spot_dict, hist_df=hist_df, index_df=index_df)
            else:
                analysis_result = analyze_all(hist_df, spot_dict)
                report = build_report(analysis_result, code, item.get("stock_name", stock_name),
                                       spot_dict, hist_df=hist_df, index_df=index_df)

            # 保存评分记录
            save_score_history({
                "stock_code": code,
                "stock_name": report["stock_name"],
                "overall_score": report["overall_score"],
                "recommendation": report["recommendation"],
                "signal_strength": report["signal_strength"],
                "action_type": report["action_type"],
                "trend_veto": report["trend_veto"],
                "factor_scores": report["scores"],
                "market_regime": report["market_regime"],
            })

            results.append({
                "stock_code": code,
                "stock_name": report["stock_name"],
                "overall_score": report["overall_score"],
                "recommendation": report["recommendation"],
                "action_type": report["action_type"],
                "signal_strength": report["signal_strength"],
                "trend_veto": report["trend_veto"],
                "summary": report["summary"],
                "risk_level": report.get("risk_management", {}).get("risk_level", "中"),
                "stop_loss": report.get("risk_management", {}).get("stop_loss", "N/A"),
                "position_pct": report.get("risk_management", {}).get("position_pct", 0),
            })
        except Exception as e:
            results.append({
                "stock_code": code, "stock_name": item.get("stock_name", code),
                "error": str(e), "overall_score": None,
                "recommendation": "分析出错", "action_type": "hold",
                "signal_strength": "无"
            })

    # 按评分排序，分类
    buy_list = [r for r in results if r.get("action_type") in ("strong_buy", "buy") and not r.get("error")]
    hold_list = [r for r in results if r.get("action_type") == "hold" and not r.get("error")]
    sell_list = [r for r in results if r.get("action_type") in ("sell", "strong_sell") and not r.get("error")]
    error_list = [r for r in results if r.get("error")]

    buy_list.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    sell_list.sort(key=lambda x: x.get("overall_score", 0))

    # 大盘环境
    try:
        index_df_scan = fetch_index_data("000001", datalen=90)
        if index_df_scan is not None and len(index_df_scan) >= 30:
            index_df_scan = calc_all_indicators(index_df_scan)
            regime, regime_desc = detect_market_regime(index_df_scan)
        else:
            regime, regime_desc = "unknown", "大盘数据不足"
    except Exception:
        regime, regime_desc = "unknown", "获取失败"

    avg_score = np.mean([r["overall_score"] for r in results if r.get("overall_score")]) \
        if any(r.get("overall_score") for r in results) else 0

    briefing = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "regime_desc": regime_desc,
        "avg_score": round(float(avg_score), 1),
        "total_count": len(results),
        "buy_count": len(buy_list),
        "hold_count": len(hold_list),
        "sell_count": len(sell_list),
        "error_count": len(error_list),
        "buy_list": buy_list[:5],
        "hold_list": hold_list[:5],
        "sell_list": sell_list[:5],
        "error_list": error_list,
        "results": results,
    }
    return briefing
