"""配置文件读写 — config.yaml（含敏感信息，不入库）。

设计要点：
- 默认配置 DEFAULT_CONFIG 提供完整字段；
- load_config 在磁盘无文件或解析失败时回退到默认值；
- 只做「段内字段合并」，不会因配置缺段而整体崩溃；
- config.yaml 已被 .gitignore 忽略，仓库仅保留 config.example.yaml 模板。
"""
import os
import yaml
from copy import deepcopy

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

DEFAULT_CONFIG = {
    "push": {
        "enabled": False,                 # 是否启用自动推送
        "feishu_webhook": "",             # 飞书自定义机器人 Webhook 地址
        "schedule": "0 * * * *",          # cron 表达式（分 时 日 月 周），默认每个整点
        "only_trading_hours": True,       # true=仅 A股交易时段推送
        "frequency_preset": "hourly_trading",  # hourly_trading | hourly | daily_preopen
    }
}


def load_config():
    """读取配置；文件不存在或解析异常时返回默认配置副本。"""
    if not os.path.exists(CONFIG_PATH):
        return deepcopy(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged = deepcopy(DEFAULT_CONFIG)
        if isinstance(data, dict):
            for sec, val in data.items():
                if isinstance(val, dict) and isinstance(merged.get(sec), dict):
                    merged[sec].update(val)
                else:
                    merged[sec] = val
        return merged
    except Exception:
        return deepcopy(DEFAULT_CONFIG)


def get_config():
    return load_config()


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def update_push_config(partial):
    """更新 push 段的部分字段并落盘。返回合并后的 push 配置 dict。

    partial 为前端传入的字段集合：
    - feishu_webhook 为空字符串/None 时不覆盖已有值（前端未改动时忽略）；
    - frequency_preset 若合法，会同时写入对应的 schedule。
    """
    cfg = load_config()
    push = cfg.setdefault("push", {})
    for k, v in partial.items():
        if k == "feishu_webhook" and (v is None or v == ""):
            continue
        push[k] = v
    save_config(cfg)
    return push
