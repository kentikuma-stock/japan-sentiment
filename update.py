#!/usr/bin/env python3
"""
日本株センチメント指数 - 毎日自動更新スクリプト
yfinanceで日経平均・日経VIを取得してindex.htmlのデータを更新する
日次データ版
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
import pytz

JST = pytz.timezone("Asia/Tokyo")
today = datetime.now(JST)
print(f"実行日時: {today.strftime('%Y-%m-%d %H:%M')} JST")

# ── 1. データ取得 ──────────────────────────────
START = "2010-11-01"
END   = today.strftime("%Y-%m-%d")

print("yfinanceからデータ取得中...")
nk_raw  = yf.download("^N225", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
jpy_raw = yf.download("JPY=X", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

vi_raw = None
for vi_ticker in ["^JNIV", "^VXJ", "JNIV"]:
    try:
        tmp = yf.download(vi_ticker, start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
        if len(tmp) > 100:
            vi_raw = tmp
            print(f"  日経VI取得成功: {vi_ticker}")
            break
    except Exception:
        continue

if vi_raw is None or len(vi_raw) < 100:
    print("  日経VI: yfinance取得不可 → 日経平均ボラティリティで代替")
    nk_ret = nk_raw.pct_change()
    vi_raw = (nk_ret.rolling(20).std() * (252**0.5) * 100).bfill()
    vi_raw = vi_raw.clip(10, 80)

print(f"  日経平均: {len(nk_raw)}件 → {nk_raw.index[-1].date()}")
print(f"  日経VI:   {len(vi_raw)}件")
print(f"  ドル円:   {len(jpy_raw)}件 → {jpy_raw.index[-1].date()}")

# ── 2. 日次に揃える（月次resampleをやめる）──────
# 共通営業日インデックスで揃える
idx = nk_raw.index.intersection(jpy_raw.index)
nk_d  = nk_raw.loc[idx]
vi_d  = vi_raw.reindex(idx, method="ffill").bfill()
jpy_d = jpy_raw.loc[idx]

# ── 3. センチメントスコア計算（日次版）──────────
def calc_scores(nk_series, vi_series, jpy_series):
    nk  = np.array(nk_series)
    vi  = np.array(vi_series)
    jpy = np.array(jpy_series)
    scores = []
    MA_WIN = 125  # 125営業日MA

    for i in range(len(nk)):
        # モメンタム（125日MA乖離）
        if i >= MA_WIN:
            ma = np.mean(nk[i-MA_WIN:i])
            dev = (nk[i] - ma) / ma * 100
            mom = max(0, min(100, 50 + dev * 2))
        else:
            mom = 50

        # VIスコア（VI高=Fear=低スコア）
        v = vi[i]
        if   v >= 60: vi_s = 5
        elif v >= 45: vi_s = 10
        elif v >= 35: vi_s = 20
        elif v >= 30: vi_s = 30
        elif v >= 25: vi_s = 40
        elif v >= 20: vi_s = 55
        elif v >= 17: vi_s = 65
        elif v >= 15: vi_s = 75
        else:         vi_s = 85

        # ドル円（円高=Fear=低スコア）
        if i >= 250:
            jpy_avg = np.mean(jpy[i-250:i])
            jpy_dev = (jpy[i] - jpy_avg) / jpy_avg * 100
            jpy_s = max(0, min(100, 50 + jpy_dev * 3))
        else:
            jpy_s = 50

        total = round(mom * 0.3 + vi_s * 0.5 + jpy_s * 0.2)
        scores.append(max(0, min(100, total)))

    return scores

scores = calc_scores(nk_d, vi_d, jpy_d)

# ── 4. TOPIXデータ ──────────────────────────────
tp_ratio = 0.078
tp_vals  = [round(v * tp_ratio) for v in nk_d.values.tolist()]

# ── 5. JSONデータ構築 ──────────────────────────
dates  = [d.strftime("%Y-%m-%d") for d in nk_d.index]  # 日次："2011-03-11"形式
nk_lst = [round(v) for v in nk_d.values.tolist()]
vi_lst = [round(v, 1) for v in vi_d.values.tolist()]

nk0, tp0 = nk_lst[0], tp_vals[0]
nk_norm = [round(v / nk0 * 100, 1) for v in nk_lst]
tp_norm = [round(v / tp0 * 100, 1) for v in tp_vals]

# イベントも正確な日付で指定
EVENTS = {
    "2011-03-11": "東日本大震災",
    "2015-08-24": "チャイナショック",
    "2016-01-21": "逆オイルショック",
    "2016-06-24": "Brexit",
    "2020-03-19": "コロナショック",
    "2022-03-09": "ウクライナ侵攻",
    "2024-08-05": "植田ショック",
    "2025-04-07": "トランプ関税",
}

latest = scores[-1]
print(f"\n最新スコア: {latest} ({dates[-1]})")
print(f"日経平均: {nk_lst[-1]:,}  VI: {vi_lst[-1]}")

# ── 6. index.htmlのデータ部分を書き換え ────────
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

def replace_js_var(html, var_name, new_value):
    pattern = rf'(const {var_name}=)\[.*?\]'
    replacement = f'\\g<1>{json.dumps(new_value)}'
    return re.sub(pattern, replacement, html, flags=re.DOTALL)

def replace_js_obj(html, var_name, new_value):
    pattern = rf'(const {var_name}=)\{{.*?\}}'
    replacement = f'\\g<1>{json.dumps(new_value, ensure_ascii=False)}'
    return re.sub(pattern, replacement, html, flags=re.DOTALL)

html = replace_js_var(html, "DATES",   dates)
html = replace_js_var(html, "NK",      nk_lst)
html = replace_js_var(html, "TP",      tp_vals)
html = replace_js_var(html, "SC",      scores)
html = replace_js_var(html, "NKN",     nk_norm)
html = replace_js_var(html, "TPN",     tp_norm)
html = replace_js_obj(html, "EVENTS",  EVENTS)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\nindex.html を更新しました ?")
print(f"データ件数: {len(dates)}営業日分 ({dates[0]} → {dates[-1]})")
