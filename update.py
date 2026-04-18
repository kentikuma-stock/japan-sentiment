#!/usr/bin/env python3
"""
日本株センチメント指数 - 毎日自動更新スクリプト
yfinanceで日経平均・日経VIを取得してindex.htmlのデータを更新する
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
START = "2010-11-01"  # 日経VI算出開始月
END   = today.strftime("%Y-%m-%d")

print("yfinanceからデータ取得中...")
nk_raw  = yf.download("^N225", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
jpy_raw = yf.download("JPY=X", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()

# 日経VIは複数ティッカーを試みる
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

# 取得できない場合は日経平均の変動率からVI代替値を計算
if vi_raw is None or len(vi_raw) < 100:
    print("  日経VI: yfinance取得不可 → 日経平均ボラティリティで代替")
    nk_daily = nk_raw.copy()
    nk_ret = nk_daily.pct_change()
    # 20日ローリング標準偏差 × √252 × 100 でVI代替値を計算
    vi_raw = (nk_ret.rolling(20).std() * (252**0.5) * 100).bfill()
    vi_raw = vi_raw.clip(10, 80)

print(f"  日経平均: {len(nk_raw)}件 ? {nk_raw.index[-1].date()}")
print(f"  日経VI:   {len(vi_raw)}件")
print(f"  ドル円:   {len(jpy_raw)}件 ? {jpy_raw.index[-1].date()}")

# ── 2. 月次に集約 ──────────────────────────────
nk_m  = nk_raw.resample("MS").last().dropna()
vi_m  = vi_raw.resample("MS").last().dropna()
jpy_m = jpy_raw.resample("MS").last().dropna()

# 共通インデックスで揃える
idx = nk_m.index.intersection(vi_m.index)
nk_m  = nk_m.loc[idx]
vi_m  = vi_m.loc[idx]
jpy_m = jpy_m.reindex(idx, method="nearest")

# ── 3. センチメントスコア計算 ──────────────────
def calc_scores(nk_series, vi_series, jpy_series):
    nk  = np.array(nk_series)
    vi  = np.array(vi_series)
    jpy = np.array(jpy_series)
    scores = []
    MA_WIN = 4  # 月次4ヶ月 ? 125日

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
        # 直近平均からの乖離で判定
        if i >= 12:
            jpy_avg = np.mean(jpy[i-12:i])
            jpy_dev = (jpy[i] - jpy_avg) / jpy_avg * 100
            jpy_s = max(0, min(100, 50 + jpy_dev * 3))
        else:
            jpy_s = 50

        total = round(mom * 0.3 + vi_s * 0.5 + jpy_s * 0.2)
        scores.append(max(0, min(100, total)))

    return scores

scores = calc_scores(nk_m, vi_m, jpy_m)

# ── 4. TOPIXデータ（日経平均に連動した推定値）──
# TOPIXはyfinance無料版で取得困難なため、日経平均と相関係数を使って推定
# 実運用では別途データソースを検討
tp_ratio = 0.078  # TOPIX ? 日経平均 × 0.078 (概算)
tp_vals  = [round(v * tp_ratio) for v in nk_m.values.tolist()]

# ── 5. JSONデータ構築 ──────────────────────────
dates  = [d.strftime("%Y-%m") for d in nk_m.index]
nk_lst = [round(v) for v in nk_m.values.tolist()]
vi_lst = [round(v, 1) for v in vi_m.values.tolist()]

nk0, tp0 = nk_lst[0], tp_vals[0]
nk_norm = [round(v / nk0 * 100, 1) for v in nk_lst]
tp_norm = [round(v / tp0 * 100, 1) for v in tp_vals]

EVENTS = {
    "2011-03": "東日本大震災",
    "2015-08": "チャイナショック",
    "2016-01": "逆オイルショック",
    "2016-06": "Brexit",
    "2020-03": "コロナショック",
    "2022-03": "ウクライナ侵攻",
    "2024-08": "植田ショック",
    "2025-04": "トランプ関税",
}

data = {
    "dates":      dates,
    "nikkei":     nk_lst,
    "topix":      tp_vals,
    "vi":         vi_lst,
    "score":      scores,
    "nikkei_norm": nk_norm,
    "topix_norm":  tp_norm,
    "events":     EVENTS,
    "updated":    today.strftime("%Y/%m/%d"),
}

latest = scores[-1]
print(f"\n最新スコア: {latest} ({dates[-1]})")
print(f"日経平均: {nk_lst[-1]:,}  VI: {vi_lst[-1]}")

# ── 6. index.htmlのデータ部分を書き換え ────────
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 各変数を置換
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
print(f"データ件数: {len(dates)}ヶ月分 ({dates[0]} ? {dates[-1]})")
