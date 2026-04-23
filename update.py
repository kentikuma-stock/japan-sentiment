#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import json
import re
import requests
from datetime import datetime
import pytz

JST = pytz.timezone("Asia/Tokyo")
today = datetime.now(JST)
print(f"実行日時: {today.strftime('%Y-%m-%d %H:%M')} JST")

START = "2010-11-01"
END   = today.strftime("%Y-%m-%d")

# ── 1. 日経平均・ドル円取得 ──────────────────────────
print("yfinanceからデータ取得中...")
nk_raw  = yf.download("^N225", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
jpy_raw = yf.download("JPY=X",  start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
print(f"  日経平均: {len(nk_raw)}件 → {nk_raw.index[-1].date()}")
print(f"  ドル円:   {len(jpy_raw)}件 → {jpy_raw.index[-1].date()}")

# ── 2. 日経VI取得（stooq直接HTTP）──────────────────
vi_raw = None
try:
    url = f"https://stooq.com/q/d/l/?s=^jniv.jp&d1=20101101&d2={today.strftime('%Y%m%d')}&i=d"
    r = requests.get(url, timeout=15)
    # カラム名を確認してから処理
    from io import StringIO
    df_vi = pd.read_csv(StringIO(r.text))
    print(f"  stooqカラム: {df_vi.columns.tolist()}")
    # 日付カラムを自動検出
    date_col = df_vi.columns[0]
    close_col = [c for c in df_vi.columns if 'Close' in c or 'close' in c][0]
    df_vi[date_col] = pd.to_datetime(df_vi[date_col])
    df_vi = df_vi.set_index(date_col).sort_index()
    vi_raw = df_vi[close_col].dropna()
    if len(vi_raw) > 100:
        print(f"  日経VI取得成功: stooq ({len(vi_raw)}件 → {vi_raw.index[-1].date()})")
    else:
        vi_raw = None
        print("  stooq: データ不足 → 代替へ")
except Exception as e:
    print(f"  stooq失敗: {e}")

if vi_raw is None:
    print("  日経VI: 代替ボラティリティで計算")
    nk_ret = nk_raw.pct_change()
    vi_raw = (nk_ret.rolling(20).std() * (252**0.5) * 100).bfill().clip(10, 80)

# ── 3. インデックス統一 ──────────────────────────────
idx   = nk_raw.index.intersection(jpy_raw.index)
nk_d  = nk_raw.loc[idx]
jpy_d = jpy_raw.loc[idx]
vi_d  = vi_raw.reindex(idx, method="ffill").bfill()
print(f"  統合後: {len(nk_d)}件 | VI最新: {vi_d.iloc[-1]:.2f}")

# ── 4. センチメントスコア計算 ─────────────────────────
def vi_to_score(v):
    if   v < 15: return 90
    elif v < 18: return 80
    elif v < 22: return 70
    elif v < 27: return 58
    elif v < 32: return 45
    elif v < 40: return 30
    elif v < 50: return 15
    else:        return 5

def calc_scores(nk_arr, vi_arr, jpy_arr):
    scores = []
    MA_WIN = 125
    for i in range(len(nk_arr)):
        if i >= MA_WIN:
            ma  = np.mean(nk_arr[i - MA_WIN:i])
            dev = (nk_arr[i] - ma) / ma * 100
            mom = max(10, min(90, 50 + dev * 2))
        else:
            mom = 50
        vi_s = vi_to_score(vi_arr[i])
        if i >= 250:
            jpy_avg = np.mean(jpy_arr[i - 250:i])
            jpy_dev = (jpy_arr[i] - jpy_avg) / jpy_avg * 100
            jpy_s   = max(0, min(100, 50 + jpy_dev * 3))
        else:
            jpy_s = 50
        total = round(mom * 0.5 + vi_s * 0.3 + jpy_s * 0.2)
        scores.append(max(0, min(100, total)))
    return scores

nk_arr  = nk_d.values.astype(float)
vi_arr  = vi_d.values.astype(float)
jpy_arr = jpy_d.values.astype(float)
scores  = calc_scores(nk_arr, vi_arr, jpy_arr)

# ── 5. データ整形 ──────────────────────────────────
dates   = [d.strftime("%Y-%m-%d") for d in nk_d.index]
nk_lst  = [round(v) for v in nk_arr.tolist()]
tp_vals = [round(v * 0.078) for v in nk_arr.tolist()]

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
print(f"  VI: {vi_arr[-1]:.2f} → VIスコア: {vi_to_score(vi_arr[-1])}")

# ── 6. HTML書き換え ────────────────────────────────
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

def replace_js_array(html, var_name, new_value):
    pattern = rf'(const {var_name}=)\[[\s\S]*?\](?=;)'
    result  = re.sub(pattern, f'\\g<1>{json.dumps(new_value)}', html)
    print(f"  {'✅' if result != html else '⚠️'} {var_name}")
    return result

def replace_js_obj(html, var_name, new_value):
    # セミコロンありなしどちらにも対応
    pattern = rf'const {var_name}=\{{[\s\S]*?\}};'
    new_str = f'const {var_name}={json.dumps(new_value, ensure_ascii=False)};'
    result  = re.sub(pattern, new_str, html)
    print(f"  {'✅' if result != html else '⚠️'} {var_name}")
    return result

print("\nindex.html 書き換え中...")
html = replace_js_array(html, "DATES",  dates)
html = replace_js_array(html, "NK",     nk_lst)
html = replace_js_array(html, "TP",     tp_vals)
html = replace_js_array(html, "SC",     scores)
html = replace_js_obj  (html, "EVENTS", EVENTS)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅ 完了: {len(dates)}営業日 ({dates[0]} → {dates[-1]})")
print(f"   最新スコア: {latest} | VI: {vi_arr[-1]:.2f}")
