#!/usr/bin/env python3
"""
日本株センチメント指数 - 毎日自動更新スクリプト
日経VI: stooq経由で取得（yfinance ^JNIVがデリスト済みのため）
"""

import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import json
import re
from datetime import datetime, timedelta
import pytz

JST = pytz.timezone("Asia/Tokyo")
today = datetime.now(JST)
print(f"実行日時: {today.strftime('%Y-%m-%d %H:%M')} JST")

START = "2010-11-01"
END   = today.strftime("%Y-%m-%d")

# ── 1. 日経平均・ドル円取得（yfinance）──────────────
print("yfinanceからデータ取得中...")
nk_raw  = yf.download("^N225", start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
jpy_raw = yf.download("JPY=X",  start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
print(f"  日経平均: {len(nk_raw)}件 → {nk_raw.index[-1].date()}")
print(f"  ドル円:   {len(jpy_raw)}件 → {jpy_raw.index[-1].date()}")

# ── 2. 日経VI取得（stooq優先、失敗時は代替）─────────
vi_raw = None

# 2-1. stooqで試みる
try:
    vi_stooq = web.DataReader("^jniv.jp", "stooq", start=START, end=END)["Close"]
    vi_stooq = vi_stooq.sort_index()  # stooqは降順で返るので昇順に
    vi_stooq.index = pd.to_datetime(vi_stooq.index)
    if len(vi_stooq) > 100:
        vi_raw = vi_stooq
        print(f"  日経VI取得成功: stooq ({len(vi_raw)}件 → {vi_raw.index[-1].date()})")
except Exception as e:
    print(f"  stooq失敗: {e}")

# 2-2. stooq失敗時はyfinanceの別ティッカーを試みる
if vi_raw is None:
    for ticker in ["^JNIV", "^VXJ", "JNIV"]:
        try:
            tmp = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)["Close"].squeeze()
            if len(tmp) > 100:
                vi_raw = tmp
                print(f"  日経VI取得成功: {ticker}")
                break
        except Exception:
            continue

# 2-3. 全部ダメなら実現ボラで代替
if vi_raw is None:
    print("  日経VI: 全ソース取得不可 → 実現ボラティリティで代替")
    nk_ret = nk_raw.pct_change()
    vi_raw = (nk_ret.rolling(20).std() * (252**0.5) * 100).bfill()
    vi_raw = vi_raw.clip(10, 80)

# ── 3. インデックス統一 ──────────────────────────────
idx    = nk_raw.index.intersection(jpy_raw.index)
nk_d   = nk_raw.loc[idx]
jpy_d  = jpy_raw.loc[idx]
vi_d   = vi_raw.reindex(idx, method="ffill").bfill()

print(f"  統合後: {len(nk_d)}営業日 ({nk_d.index[0].date()} → {nk_d.index[-1].date()})")
print(f"  最新VI: {vi_d.iloc[-1]:.2f}")

# ── 4. センチメントスコア計算 ─────────────────────────
#
# 指標:
#   日経平均モメンタム（125日MA乖離） × 0.5
#   日経VI反転スコア                  × 0.3
#   ドル円スコア                      × 0.2
#
# VI閾値（日経VIは米VIXより高め推移。通常15?25、警戒30超、パニック50超）
def vi_to_score(v):
    if   v < 15: return 90
    elif v < 18: return 80
    elif v < 22: return 70
    elif v < 27: return 58  # 平常?やや安心
    elif v < 32: return 45  # やや警戒（現在VI?29.73はここ）
    elif v < 40: return 30  # 警戒
    elif v < 50: return 15  # 強い警戒
    else:        return 5   # パニック

def calc_scores(nk_arr, vi_arr, jpy_arr):
    scores = []
    MA_WIN = 125

    for i in range(len(nk_arr)):
        # モメンタム
        if i >= MA_WIN:
            ma  = np.mean(nk_arr[i - MA_WIN:i])
            dev = (nk_arr[i] - ma) / ma * 100
            mom = max(10, min(90, 50 + dev * 2))
        else:
            mom = 50

        # VIスコア
        vi_s = vi_to_score(vi_arr[i])

        # ドル円スコア
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
tp_vals = [round(v * 0.078) for v in nk_arr.tolist()]  # TOPIX固定比率

nk0    = nk_lst[0]
tp0    = tp_vals[0]
nk_norm = [round(v / nk0 * 100, 1) for v in nk_lst]
tp_norm = [round(v / tp0 * 100, 1) for v in tp_vals]

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
vi_now = vi_arr[-1]
ma125  = np.mean(nk_arr[-126:-1])
dev    = (nk_arr[-1] - ma125) / ma125 * 100
mom    = max(10, min(90, 50 + dev * 2))

print(f"\n最新スコア: {latest} ({dates[-1]})")
print(f"  日経平均: {nk_lst[-1]:,}  125日MA: {ma125:.0f}  乖離: {dev:+.1f}%")
print(f"  モメンタムスコア: {mom:.0f}")
print(f"  VI: {vi_now:.2f}  VIスコア: {vi_to_score(vi_now)}")

# ── 6. HTML書き換え ────────────────────────────────
with open("index.html", "r", encoding="utf-8") as f:
    html = f.read()

def replace_js_array(html, var_name, new_value):
    pattern = rf'(const {var_name}=)\[[\s\S]*?\](?=;)'
    result  = re.sub(pattern, f'\\g<1>{json.dumps(new_value)}', html)
    status  = "?" if result != html else "?? 置換失敗"
    print(f"  {status} {var_name}")
    return result

def replace_js_obj(html, var_name, new_value):
    pattern = rf'(const {var_name}=)\{{[\s\S]*?\}}(?=;)'
    result  = re.sub(pattern, f'\\g<1>{json.dumps(new_value, ensure_ascii=False)}', html)
    status  = "?" if result != html else "?? 置換失敗"
    print(f"  {status} {var_name}")
    return result

print("\nindex.html 書き換え中...")
html = replace_js_array(html, "DATES",  dates)
html = replace_js_array(html, "NK",     nk_lst)
html = replace_js_array(html, "TP",     tp_vals)
html = replace_js_array(html, "SC",     scores)
html = replace_js_obj  (html, "EVENTS", EVENTS)

# NKN・TPNはHTMLのmap()で自動計算されるので書き換え不要

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n? 完了: {len(dates)}営業日 ({dates[0]} → {dates[-1]})")
print(f"   最新スコア: {latest} | VI: {vi_now:.2f} | VIスコア: {vi_to_score(vi_now)}")
