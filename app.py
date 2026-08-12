"""
股票信号判断系统 v4.1 - Flask 后端
正交因子模型 + 回测验证 + 市场环境感知 + 风险管理 + 交易闭环 + 智能预警
v4.1: 持仓单向数据流(交易驱动) + 日期格式统一
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response

from data_fetcher import fetch_stock_data, fetch_hist_data, fetch_index_data, is_etf, fetch_spot_data
from indicators import analyze_all, analyze_etf, calc_all_indicators
from scorer import build_report, build_etf_report, detect_market_regime, run_backtest, score_to_label
from db import (save_score_history, save_backtest_result, load_backtest_result,
                load_score_history, add_watchlist, remove_watchlist, list_watchlist,
                get_conn,
                list_holdings,
                list_trade_journal, delete_trade_journal,
                add_alert_rule, list_alert_rules, update_alert_status, delete_alert_rule)
from portfolio import get_portfolio_overview
from trade_journal import log_trade, get_trade_journal_summary, calc_signal_hit_rate
from alert_checker import check_all_alerts, generate_alert_summary, get_alert_type_label

class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app = Flask(__name__)


def format_spot_value(val):
    """格式化 spot 数据中的数值"""
    if val is None or val == "-" or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    try:
        f = float(val)
        if abs(f) >= 1e8:
            return f"{f / 1e8:.2f}亿"
        elif abs(f) >= 1e4:
            return f"{f / 1e4:.2f}万"
        else:
            return f"{f:.2f}"
    except (ValueError, TypeError):
        return str(val)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze")
def analyze():
    stock_code = request.args.get("code", "").strip()
    period = request.args.get("period", "3m")

    if not stock_code:
        return jsonify({"error": "请输入股票代码"}), 400

    if len(stock_code) != 6 or not stock_code.isdigit():
        return jsonify({"error": "股票代码应为6位数字，如 600519"}), 400

    try:
        hist_df, spot_dict, stock_name, index_df = fetch_stock_data(stock_code, period)

        if hist_df.empty or len(hist_df) < 30:
            return jsonify({"error": f"股票 {stock_code} 历史数据不足（至少需要30天数据）"}), 400

        # 计算大盘指数指标
        if index_df is not None and len(index_df) >= 30:
            index_df = calc_all_indicators(index_df)

        # 根据ETF/个股路由不同分析路径
        etf_flag = is_etf(stock_code)
        if etf_flag:
            analysis_result = analyze_etf(hist_df, spot_dict)
            report = build_etf_report(analysis_result, stock_code, stock_name, spot_dict,
                                      hist_df=hist_df, index_df=index_df)
        else:
            analysis_result = analyze_all(hist_df, spot_dict)
            report = build_report(analysis_result, stock_code, stock_name, spot_dict,
                                  hist_df=hist_df, index_df=index_df)

        # 格式化数值
        report["current_price"] = format_spot_value(report["current_price"])
        report["change_pct"] = format_spot_value(report["change_pct"])
        for k, v in report["spot_data"].items():
            report["spot_data"][k] = format_spot_value(v)

        # 保存评分记录到数据库
        save_score_history({
            "stock_code": stock_code,
            "stock_name": stock_name,
            "overall_score": report["overall_score"],
            "recommendation": report["recommendation"],
            "signal_strength": report["signal_strength"],
            "action_type": report["action_type"],
            "trend_veto": report["trend_veto"],
            "factor_scores": report["scores"],
            "market_regime": report["market_regime"],
        })

        # 保存回测结果
        if report["backtest"]["total_signals"] > 0:
            save_backtest_result({
                "stock_code": stock_code,
                "period": period,
                "model_version": report["backtest"].get("model_version", "v2"),
                "start_date": report["backtest"].get("start_date", ""),
                "end_date": report["backtest"].get("end_date", ""),
                "total_signals": report["backtest"]["total_signals"],
                "win_rate": report["backtest"]["win_rate"],
                "avg_return": report["backtest"]["avg_return"],
                "max_drawdown": report["backtest"]["max_drawdown"],
                "sharpe_ratio": report["backtest"]["sharpe_ratio"],
                "profit_factor": report["backtest"]["profit_factor"],
                "results_json": {},
            })

        # 添加K线数据用于前端图表渲染
        chart_data = prepare_chart_data(hist_df)
        report["chart_data"] = chart_data

        return Response(json.dumps(report, cls=NumpyEncoder, ensure_ascii=False),
                        mimetype='application/json')

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"分析出错: {str(e)}"}), 500


@app.route("/api/backtest")
def backtest():
    """单独的回测API，可以指定更长的时间范围"""
    stock_code = request.args.get("code", "").strip()
    period = request.args.get("period", "1y")

    if not stock_code:
        return jsonify({"error": "请输入股票代码"}), 400

    if len(stock_code) != 6 or not stock_code.isdigit():
        return jsonify({"error": "股票代码应为6位数字"}), 400

    try:
        # 回测用1年数据
        datalen_map = {"3m": 90, "6m": 180, "1y": 365}
        datalen = datalen_map.get(period, 365)

        hist_df = fetch_hist_data(stock_code, period, use_cache=True)
        spot_dict = fetch_spot_data(stock_code)

        if hist_df.empty or len(hist_df) < 60:
            return jsonify({"error": "数据不足，至少需要60天数据"}), 400

        result = run_backtest(hist_df, spot_dict)

        # 检查是否有缓存结果
        cached = load_backtest_result(stock_code, period)
        if cached and cached.get("total_signals", 0) >= result.get("total_signals", 0):
            return jsonify(cached)

        save_backtest_result({
            "stock_code": stock_code,
            "period": period,
            **result,
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"回测出错: {str(e)}"}), 500


@app.route("/api/history")
def history():
    """获取某股票的历史评分记录"""
    stock_code = request.args.get("code", "").strip()
    limit = int(request.args.get("limit", "10"))

    if not stock_code:
        return jsonify({"error": "请输入股票代码"}), 400

    records = load_score_history(stock_code, limit)
    return jsonify(records)


@app.route("/api/market_regime")
def market_regime():
    """获取当前大盘市场环境"""
    try:
        index_df = fetch_index_data("000001", datalen=90)
        if index_df is None or len(index_df) < 30:
            return jsonify({"regime": "unknown", "desc": "大盘数据不足"})

        index_df = calc_all_indicators(index_df)
        regime, desc = detect_market_regime(index_df)

        return jsonify({"regime": regime, "desc": desc})
    except Exception as e:
        return jsonify({"regime": "unknown", "desc": f"获取失败: {str(e)}"})


@app.route("/api/watchlist", methods=["GET", "POST", "DELETE"])
def watchlist():
    """自选股管理: GET列表 / POST添加 / DELETE删除"""
    if request.method == "GET":
        items = list_watchlist()
        return Response(json.dumps(items, cls=NumpyEncoder, ensure_ascii=False),
                        mimetype='application/json')

    elif request.method == "POST":
        stock_code = request.json.get("code", "").strip() if request.is_json else request.form.get("code", "").strip()
        if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
            return jsonify({"error": "股票代码应为6位数字"}), 400

        # 尝试获取股票名称
        stock_name = ""
        try:
            from data_fetcher import fetch_spot_data
            spot = fetch_spot_data(stock_code)
            stock_name = spot.get("名称", stock_code)
        except Exception:
            stock_name = stock_code

        add_watchlist(stock_code, stock_name)
        return jsonify({"ok": True, "code": stock_code, "name": stock_name})

    elif request.method == "DELETE":
        stock_code = request.args.get("code", "").strip()
        if not stock_code:
            return jsonify({"error": "请提供股票代码"}), 400
        remove_watchlist(stock_code)
        return jsonify({"ok": True})


@app.route("/api/daily_scan")
def daily_scan():
    """扫描全部自选股，生成每日投资简报"""
    items = list_watchlist()
    if not items:
        return jsonify({"error": "自选股列表为空，请先添加自选股"}), 400

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

    avg_score = np.mean([r["overall_score"] for r in results if r.get("overall_score")]) if any(r.get("overall_score") for r in results) else 0

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

    return Response(json.dumps(briefing, cls=NumpyEncoder, ensure_ascii=False),
                    mimetype='application/json')


@app.route("/api/portfolio_dashboard")
def portfolio_dashboard():
    """自选股组合仪表盘 — 行业分散度、信号分布、组合风险"""
    items = list_watchlist()
    if not items:
        return jsonify({"error": "自选股列表为空"}), 400

    # 收集所有自选股的评分记录
    scores_list = []
    risk_levels = []
    action_types = []
    stock_names = []

    for item in items:
        code = item["stock_code"]
        # 从最近评分记录中获取数据
        history = load_score_history(code, limit=1)
        if history:
            latest = history[0]
            scores_list.append(latest.get("overall_score", 50))
            factor_scores = latest.get("factor_scores", {})
            if isinstance(factor_scores, str):
                try:
                    factor_scores = json.loads(factor_scores)
                except Exception:
                    factor_scores = {}
            risk_levels.append(item.get("risk_level", "中"))
            action_types.append(latest.get("action_type", "hold"))
            stock_names.append(item.get("stock_name", code))
        else:
            scores_list.append(50)
            risk_levels.append("中")
            action_types.append("hold")
            stock_names.append(item.get("stock_name", code))

    avg_score = np.mean(scores_list) if scores_list else 50

    # 信号分布
    buy_count = sum(1 for a in action_types if a in ("strong_buy", "buy"))
    hold_count = sum(1 for a in action_types if a == "hold")
    sell_count = sum(1 for a in action_types if a in ("sell", "strong_sell"))

    # 风险等级分布
    high_risk = sum(1 for r in risk_levels if r == "高")
    medium_risk = sum(1 for r in risk_levels if r == "中")
    low_risk = sum(1 for r in risk_levels if r == "低")

    # 组合风险等级 (按加权)
    if high_risk > len(items) * 0.5:
        portfolio_risk = "高"
    elif high_risk > 0 or medium_risk > len(items) * 0.5:
        portfolio_risk = "中"
    else:
        portfolio_risk = "低"

    # 行业/类型分散度
    industry_map = {
        "60": "主板", "00": "深市主板", "30": "创业板",
        "68": "科创板",
        "51": "ETF基金", "56": "ETF基金", "58": "ETF基金",
        "50": "ETF基金", "15": "ETF基金", "16": "ETF基金",
    }
    industries = {}
    for item in items:
        code = item["stock_code"]
        prefix = code[:2]
        ind_name = industry_map.get(prefix, "其他")
        industries[ind_name] = industries.get(ind_name, 0) + 1

    concentration_warning = ""
    max_ind = max(industries.values()) if industries else 0
    if max_ind >= len(items) * 0.7 and len(items) >= 3:
        concentration_warning = "⚠️ 自选股过度集中在同一市场板块，建议分散配置降低风险"

    dashboard = {
        "total_stocks": len(items),
        "avg_score": round(float(avg_score), 1),
        "score_label": score_to_label(float(avg_score)),
        "signal_distribution": {"buy": buy_count, "hold": hold_count, "sell": sell_count},
        "risk_distribution": {"high": high_risk, "medium": medium_risk, "low": low_risk},
        "portfolio_risk": portfolio_risk,
        "industries": industries,
        "concentration_warning": concentration_warning,
        "stock_details": [
            {"name": n, "score": s, "action": a, "risk": r}
            for n, s, a, r in zip(stock_names, scores_list, action_types, risk_levels)
        ],
    }

    return Response(json.dumps(dashboard, cls=NumpyEncoder, ensure_ascii=False),
                    mimetype='application/json')


def prepare_chart_data(df):
    """将K线DataFrame转换为前端ECharts所需的格式"""
    chart_data = {
        "dates": [],
        "kline": [],  # [open, close, low, high]
        "volumes": [],
        "ma5": [],
        "ma10": [],
        "ma20": [],
        "ma60": [],
        "boll_upper": [],
        "boll_mid": [],
        "boll_lower": [],
        "dif": [],
        "dea": [],
        "macd": [],
        "rsi": [],
        "mfi": [],
        "obv": [],
        "adx": [],
        "atr": [],
        "sar": [],
    }

    for _, row in df.iterrows():
        date_str = row["日期"].strftime("%Y-%m-%d") if isinstance(row["日期"], pd.Timestamp) else str(row["日期"])
        chart_data["dates"].append(date_str)
        chart_data["kline"].append([
            round(row.get("开盘", 0), 2),
            round(row.get("收盘", 0), 2),
            round(row.get("最低", 0), 2),
            round(row.get("最高", 0), 2),
        ])
        chart_data["volumes"].append(round(row.get("成交量", 0), 0))

        # 各指标数据 (NaN用null表示)
        for field in ["ma5", "ma10", "ma20", "ma60", "boll_upper", "boll_mid", "boll_lower",
                       "dif", "dea", "macd", "rsi", "mfi", "obv", "adx", "atr", "sar"]:
            val = row.get(field, None)
            if val is not None and not pd.isna(val):
                chart_data[field].append(round(float(val), 4))
            else:
                chart_data[field].append(None)

    return chart_data


# ============================================================
# 持仓管理路由
# ============================================================

@app.route("/portfolio")
def portfolio_page():
    """持仓管理独立页面"""
    return render_template("portfolio.html")


@app.route("/api/portfolio/overview")
def portfolio_overview():
    """获取持仓组合总览（实时行情+盈亏+建议）"""
    try:
        overview = get_portfolio_overview()
        return Response(json.dumps(overview, cls=NumpyEncoder, ensure_ascii=False),
                        mimetype='application/json')
    except Exception as e:
        return jsonify({"error": f"获取持仓总览失败: {str(e)}"}), 500


@app.route("/api/portfolio/holdings", methods=["GET"])
def holdings():
    """持仓列表(只读) — 持仓只能通过交易记录创建和更新"""
    items = list_holdings()
    return Response(json.dumps(items, cls=NumpyEncoder, ensure_ascii=False),
                    mimetype='application/json')


# ============================================================
# 交易日志路由
# ============================================================

@app.route("/journal")
def journal_page():
    """交易日志独立页面"""
    return render_template("journal.html")


@app.route("/api/trade_journal", methods=["GET", "POST", "DELETE"])
def trade_journal_api():
    """交易日志 CRUD"""
    if request.method == "GET":
        stock_code = request.args.get("code", "").strip()
        if stock_code:
            items = list_trade_journal(stock_code=stock_code, limit=200)
        else:
            items = list_trade_journal(limit=200)
        return Response(json.dumps(items, cls=NumpyEncoder, ensure_ascii=False),
                        mimetype='application/json')

    elif request.method == "POST":
        data = request.json if request.is_json else request.form
        stock_code = (data.get("code") or data.get("stock_code") or "").strip()
        action_type = (data.get("action_type") or "").strip()
        quantity = data.get("quantity")
        price = data.get("price")
        reason = (data.get("reason") or "").strip()
        follow_signal = data.get("follow_signal", True)
        trade_date = (data.get("trade_date") or "").strip() or None

        if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
            return jsonify({"error": "股票代码应为6位数字"}), 400
        if action_type not in ("buy", "sell"):
            return jsonify({"error": "操作类型必须是 buy 或 sell"}), 400
        if not quantity or int(quantity) <= 0:
            return jsonify({"error": "数量必须大于0"}), 400
        if not price or float(price) <= 0:
            return jsonify({"error": "价格必须大于0"}), 400

        # 获取股票名称
        stock_name = ""
        try:
            spot = fetch_spot_data(stock_code)
            stock_name = spot.get("名称", stock_code)
        except Exception:
            stock_name = stock_code

        # log_trade 内部会自动同步更新持仓(单向: 交易 → 持仓)
        result = log_trade(
            stock_code, stock_name, action_type,
            int(quantity), float(price),
            reason=reason,
            follow_signal=follow_signal,
            trade_date=trade_date,
        )
        return jsonify(result)

    elif request.method == "DELETE":
        journal_id = request.args.get("id", "")
        if not journal_id:
            return jsonify({"error": "请提供日志ID"}), 400
        delete_trade_journal(int(journal_id))
        return jsonify({"ok": True})


@app.route("/api/trade_journal/summary")
def trade_journal_summary():
    """交易日志汇总"""
    summary = get_trade_journal_summary()
    return Response(json.dumps(summary, cls=NumpyEncoder, ensure_ascii=False),
                    mimetype='application/json')


@app.route("/api/signal_hit_rate")
def signal_hit_rate():
    """信号命中率分析"""
    days_back = int(request.args.get("days_back", "30"))
    forward_days = int(request.args.get("forward_days", "5"))

    result = calc_signal_hit_rate(days_back=days_back, forward_days=forward_days)
    return Response(json.dumps(result, cls=NumpyEncoder, ensure_ascii=False),
                    mimetype='application/json')


# ============================================================
# 预警管理路由
# ============================================================

@app.route("/alerts")
def alerts_page():
    """预警管理独立页面"""
    return render_template("alerts.html")


@app.route("/api/alerts", methods=["GET", "POST", "DELETE"])
def alerts_api():
    """预警规则 CRUD"""
    if request.method == "GET":
        status = request.args.get("status", "")
        items = list_alert_rules(status=status if status else None)
        # 添加类型标签
        for item in items:
            item["type_label"] = get_alert_type_label(item["alert_type"])
        return Response(json.dumps(items, cls=NumpyEncoder, ensure_ascii=False),
                        mimetype='application/json')

    elif request.method == "POST":
        data = request.json if request.is_json else request.form
        stock_code = (data.get("code") or data.get("stock_code") or "").strip()
        alert_type = (data.get("alert_type") or "").strip()
        threshold = data.get("threshold")
        message = (data.get("message") or "").strip()

        if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
            return jsonify({"error": "股票代码应为6位数字"}), 400

        valid_types = ("price_above", "price_below", "signal_buy", "signal_sell",
                       "stop_loss", "take_profit", "daily_drop", "daily_rise")
        if alert_type not in valid_types:
            return jsonify({"error": f"预警类型无效，可选: {', '.join(valid_types)}"}), 400

        # 获取股票名称
        stock_name = ""
        try:
            spot = fetch_spot_data(stock_code)
            stock_name = spot.get("名称", stock_code)
        except Exception:
            stock_name = stock_code

        threshold_val = float(threshold) if threshold else None

        add_alert_rule(stock_code, stock_name, alert_type, threshold_val, message)
        return jsonify({"ok": True, "code": stock_code, "name": stock_name,
                        "alert_type": alert_type, "type_label": get_alert_type_label(alert_type)})

    elif request.method == "DELETE":
        rule_id = request.args.get("id", "")
        if not rule_id:
            return jsonify({"error": "请提供预警规则ID"}), 400
        delete_alert_rule(int(rule_id))
        return jsonify({"ok": True})


@app.route("/api/alerts/check", methods=["POST"])
def alerts_check():
    """手动触发预警检查"""
    result = check_all_alerts()
    return Response(json.dumps(result, cls=NumpyEncoder, ensure_ascii=False),
                    mimetype='application/json')


@app.route("/api/alerts/summary")
def alerts_summary():
    """预警汇总(用于首页速览)"""
    result = generate_alert_summary()
    return Response(json.dumps(result, cls=NumpyEncoder, ensure_ascii=False),
                    mimetype='application/json')


@app.route("/api/alerts/<int:rule_id>/reset", methods=["POST"])
def alert_reset(rule_id):
    """重置已触发的预警为活跃状态"""
    update_alert_status(rule_id, "active")
    return jsonify({"ok": True})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
