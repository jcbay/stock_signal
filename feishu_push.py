"""飞书推送模块 — 把简报渲染为飞书卡片并发送。

设计取舍（按用户决策）：
- 采用确定性模板渲染，不接入 LLM（无需 API key，稳定可复现）；
- 推送频率为「A股交易时段的每个整点」（可在配置/设置页调整为每小时或盘前）；
- 配置写在 config.yaml（含 webhook，已 gitignore）；
- 调度器用 APScheduler，在 app.py 启动时拉起；
- 暂不加多实例锁：单进程部署（本地 / Render workers=1）下无重复推送问题，
  多 worker 部署会重复推送，后续可加文件锁/DB 锁解决。
"""
import time
import requests
from datetime import datetime, timezone, timedelta

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APS = True
except Exception:
    HAS_APS = False

from config import get_config

# 上海时区（中国无夏令时，固定 +8）
SHANGHAI = timezone(timedelta(hours=8))

# A股交易时段：上午 9:30-11:30，下午 13:00-15:00
TRADING_WINDOWS = [("09:30", "11:30"), ("13:00", "15:00")]

# 频率预设 → (cron, only_trading_hours)
FREQUENCY_PRESETS = {
    "hourly_trading": {"schedule": "0 * * * *", "only_trading_hours": True},
    "hourly":         {"schedule": "0 * * * *", "only_trading_hours": False},
    "daily_preopen":  {"schedule": "26 9 * * 1-5", "only_trading_hours": False},
}

_scheduler = None


def now_shanghai():
    return datetime.now(timezone.utc).astimezone(SHANGHAI)


def is_trading_day(dt=None):
    dt = dt or now_shanghai()
    return dt.weekday() < 5  # 0-4 = 周一至周五


def is_trading_time(dt=None):
    """是否为交易日内的交易时段（仅判断时段，不判断整点）。"""
    dt = dt or now_shanghai()
    if dt.weekday() >= 5:
        return False
    now_min = dt.hour * 60 + dt.minute
    for start, end in TRADING_WINDOWS:
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        if sh * 60 + sm <= now_min <= eh * 60 + em:
            return True
    return False


def render_feishu_card(briefing):
    """把 briefing dict 渲染为飞书 interactive 卡片（dict）。"""
    date = briefing.get("date") or now_shanghai().strftime("%Y-%m-%d %H:%M")
    regime = briefing.get("regime", "unknown")
    regime_desc = briefing.get("regime_desc", "")
    avg = briefing.get("avg_score", 0)
    regime_label = {"bullish": "多头", "bearish": "空头",
                    "sideways": "震荡", "unknown": "未知"}.get(regime, regime)
    header_color = {"bullish": "green", "sideways": "blue",
                    "bearish": "orange", "unknown": "grey"}.get(regime, "blue")

    summary = (
        f"大盘环境：{regime_label}（{regime_desc}）\n"
        f"平均评分：{avg}　自选股：{briefing.get('total_count', 0)} 支\n"
        f"买入 {briefing.get('buy_count', 0)} / 持有 {briefing.get('hold_count', 0)} "
        f"/ 卖出 {briefing.get('sell_count', 0)}"
        + (f"　分析失败 {briefing.get('error_count', 0)}" if briefing.get("error_count") else "")
    )

    elements = [
        {"tag": "div", "text": {"tag": "lark_md", "content": summary}},
        {"tag": "hr"},
    ]

    def list_block(title, items):
        if not items:
            return
        lines = [f"**{title}（{len(items)}）**"]
        for it in items:
            name = it.get("stock_name") or it.get("stock_code")
            code = it.get("stock_code")
            score = it.get("overall_score")
            rec = it.get("recommendation", "")
            risk = it.get("risk_level", "")
            stop = it.get("stop_loss", "")
            line = f"- {name}({code}) 评分 {score}｜{rec}"
            if risk:
                line += f"｜风险 {risk}"
            if stop and stop != "N/A":
                line += f"｜止损 {stop}"
            lines.append(line)
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}})

    list_block("买入候选", briefing.get("buy_list", []))
    list_block("持有观察", briefing.get("hold_list", []))
    list_block("卖出警示", briefing.get("sell_list", []))

    if briefing.get("error_list"):
        errs = briefing["error_list"]
        lines = ["**分析失败**"] + [
            f"- {e.get('stock_name') or e.get('stock_code')}：{e.get('error', '')}" for e in errs
        ]
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}})

    elements.append({"tag": "hr"})
    elements.append({"tag": "note",
                     "content": "本简报由股票信号系统自动生成，仅供研究参考，不构成投资建议。"})

    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"A股信号简报 · {date}"},
                "template": header_color,
            },
            "elements": elements,
        },
    }


def send_to_feishu(payload, webhook, timeout=10, retries=2):
    """发送 payload 到飞书 webhook，失败重试。返回 (success, message)。"""
    last_err = "未知错误"
    for attempt in range(retries + 1):
        try:
            resp = requests.post(webhook, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # 飞书自定义机器人成功时返回 {"code":0,"msg":"success",...}
                if data.get("code") == 0 or data.get("StatusMessage") == "success":
                    return True, "发送成功"
                last_err = f"飞书返回错误: code={data.get('code')}, msg={data.get('msg')}"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(2)
    return False, last_err


def push_briefing_now():
    """生成最新简报并推送到飞书。返回 (success, message)。"""
    from briefing import build_briefing
    briefing = build_briefing()
    if briefing is None:
        return False, "自选股列表为空，无法生成简报"
    cfg = get_config()
    webhook = (cfg.get("push", {}).get("feishu_webhook") or "").strip()
    if not webhook:
        return False, "未配置飞书 webhook"
    card = render_feishu_card(briefing)
    return send_to_feishu(card, webhook)


def send_test_message(webhook):
    """发送一条测试消息，验证 webhook 是否可用。"""
    if not webhook or not webhook.strip():
        return False, "未配置飞书 webhook"
    card = {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {"title": {"tag": "plain_text", "content": "股票信号系统 · 推送测试"},
                       "template": "blue"},
            "elements": [
                {"tag": "div", "text": {"tag": "lark_md",
                 "content": "这是一条来自股票信号系统的测试消息。\n如果你在飞书群里收到它，说明推送配置已生效。"}},
                {"tag": "note", "content": "测试时间：" + now_shanghai().strftime("%Y-%m-%d %H:%M:%S")},
            ],
        },
    }
    return send_to_feishu(card, webhook.strip())


def scheduled_push():
    """调度器回调：检查开关与交易时段后再推送。"""
    cfg = get_config()
    push = cfg.get("push", {})
    if not push.get("enabled", False):
        return
    if push.get("only_trading_hours", True) and not is_trading_time():
        return
    push_briefing_now()


# ---------------- 调度器管理 ----------------
def setup_scheduler():
    """启动时建立调度器（始终运行；是否推送由 scheduled_push 内部判断开关与时段）。"""
    global _scheduler
    if not HAS_APS:
        print("[push] 未安装 apscheduler，自动推送调度不可用")
        return
    if _scheduler is None:
        _scheduler = BackgroundScheduler(timezone=SHANGHAI)
    else:
        _scheduler.remove_all_jobs()
    cron = get_config().get("push", {}).get("schedule", "0 * * * *")
    try:
        parts = cron.split()
        if len(parts) != 5:
            raise ValueError("cron 表达式需为 5 段")
        _scheduler.add_job(scheduled_push, CronTrigger(
            minute=parts[0], hour=parts[1], day=parts[2],
            month=parts[3], day_of_week=parts[4], timezone=SHANGHAI))
        if not _scheduler.running:
            _scheduler.start()
        print(f"[push] 调度器已启动, cron='{cron}' (Asia/Shanghai)")
    except Exception as e:
        print(f"[push] 调度器启动失败: {e}")


def reschedule():
    """配置变更后重建任务。"""
    setup_scheduler()
