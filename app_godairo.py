import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import re
import requests

# --- ページ設定 ---
st.set_page_config(page_title="🔱大黒天AI V6", layout="wide")
url_input = st.sidebar.text_input("① レースのURL (オッズ取得用)", "")
st.sidebar.markdown("② 出馬表を「適当に全部コピー」してペースト")
pasted_text = st.sidebar.text_area("", height=150)

st.sidebar.markdown("---")
budget = st.sidebar.number_input("💸 今回の軍資金 (円)", value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.markdown("🌤️ レース条件（手動入力）")
course_type = st.sidebar.selectbox("コース", ["芝", "ダート", "障害"])
distance = st.sidebar.number_input("距離(m)", value=1600, step=100)
weather = st.sidebar.selectbox("天候", ["晴", "曇", "小雨", "雨", "雪"])
ground = st.sidebar.selectbox("馬場", ["良", "稍重", "重", "不良"])

course_num = {"芝": 0, "ダート": 1, "障害": 2}[course_type]
weather_num = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}[weather]
ground_num = {"良": 0, "稍重": 1, "重": 2, "不良": 3}[ground]

if st.sidebar.button("🔱 予想・ケリー計算を実行"):
    if not models:
        st.error("モデルファイルが見つかりません。")
        st.stop()
        
    if not pasted_text.strip():
        st.warning("⚠️ テキスト枠に出馬表をコピペしてください。")
        st.stop()
        
    m = re.search(r'race_id=(\d+)', url_input)
    race_id = m.group(1) if m else url_input.strip()
        
    with st.spinner("🏇 データを抽出＆最新オッズを取得中..."):
        df_race = parse_pasted_text(pasted_text)
        real_odds = fetch_real_odds(race_id) if race_id else {'win': {}, 'place': {}}
        
    if df_race is None or df_race.empty:
        st.error("❌ 抽出不可能なレベルのテキストです。もう一度コピペしてみてください。")
    else:
        df_race['単勝オッズ'] = df_race['馬番'].map(lambda x: real_odds['win'].get(x, 10.0))
        df_race['複勝オッズ_下限'] = df_race['馬番'].map(lambda x: real_odds['place'].get(x, 1.1))
        
        st.success(f"✅ 【抽出成功】{len(df_race)}頭の馬名を発見し、最新オッズを結合しました！")
        
        df_pred = df_race.copy()
        df_pred['コース_num'] = course_num
        df_pred['距離'] = distance
        df_pred['天候_num'] = weather_num
        df_pred['馬場状態_num'] = ground_num
        df_pred['斤量_num'] = 55.0
        df_pred['体重'] = 480.0
        df_pred['年齢'] = 3.0
        df_pred['出走間隔_days'] = 60.0
        df_pred['騎手_num'] = -1
        df_pred['調教師_num'] = -1
        
        mem_cols = ['全体過去平均着順', '前走着順', '出走回数', '前走脚質_num', '前走上がり3F', '過去平均上がり3F']
        for col in mem_cols:
            if not memory_df.empty and col in memory_df.columns:
                df_pred[col] = df_pred['馬名'].map(lambda x: memory_df.loc[x, col] if x in memory_df.index else np.nan)
            else:
                df_pred[col] = np.nan
                
        df_pred['全体過去平均着順'] = df_pred['全体過去平均着順'].fillna(8.0)
        df_pred['前走着順'] = df_pred['前走着順'].fillna(8.0)
        df_pred['出走回数'] = df_pred['出走回数'].fillna(1.0)
        df_pred['前走脚質_num'] = df_pred['前走脚質_num'].fillna(4.0)
        df_pred['前走上がり3F'] = df_pred['前走上がり3F'].fillna(35.0)
        df_pred['過去平均上がり3F'] = df_pred['過去平均上がり3F'].fillna(35.0)
        
        features = [
            '枠番', '単勝オッズ', '斤量_num', 'コース_num', '距離', '天候_num', 
            '体重', '年齢', '馬場状態_num', '騎手_num', '調教師_num', '出走間隔_days',
            '全体過去平均着順', '前走着順', '出走回数', '前走脚質_num', '前走上がり3F', '過去平均上がり3F'
        ]
        
        X_pred = df_pred[features].fillna(0)
        
        for name, model in models.items():
            preds = model.predict(X_pred)
            win_prob = preds / (preds.sum() if preds.sum() > 0 else 1)
            df_race[name] = np.clip(1 - (1 - win_prob) ** 2.85, 0, 1)
            
        model_cols = ['徳川家康予測', '前田利家予測', '上杉景勝予測', '毛利輝元予測', '宇喜多秀家予測']
        
        df_race['AI複勝率_num'] = df_race[model_cols].mean(axis=1)
        df_race['複勝期待値(EV)'] = df_race['AI複勝率_num'] * df_race['複勝オッズ_下限']
        
        # 💸 ケリー基準の計算（実戦向けハーフ・ケリー）
        # f* = (p * b - q) / b (p=勝率, b=オッズ-1, q=負け率)
        df_race['b_value'] = df_race['複勝オッズ_下限'] - 1.0
        df_race['ケリー割合'] = np.where(
            df_race['b_value'] > 0, 
            (df_race['AI複勝率_num'] * df_race['b_value'] - (1.0 - df_race['AI複勝率_num'])) / df_race['b_value'], 
            0
        )
        df_race['ケリー割合'] = np.clip(df_race['ケリー割合'], 0, 1) # 0〜100%に収める
        
        # ハーフケリー（推奨額の半分）を採用し、予算を掛ける
        df_race['推奨金額(円)'] = (budget * df_race['ケリー割合'] * 0.5).astype(int)
        df_race['推奨金額(円)'] = (df_race['推奨金額(円)'] // 100) * 100 # 100円単位に切り捨て
        
        def get_rank(ev, amt):
            if ev >= 1.20 and amt > 0: return "👑 勝負"
            elif ev >= 1.05 and amt > 0: return "🔥 狙い"
            return "見送り"
            
        df_race['おすすめ度'] = df_race.apply(lambda row: get_rank(row['複勝期待値(EV)'], row['推奨金額(円)']), axis=1)
        
        df_race['AI複勝率'] = (df_race['AI複勝率_num'] * 100).map('{:.1f}%'.format)
        for col in model_cols:
            df_race[col] = (df_race[col] * 100).map('{:.1f}%'.format)
            
        st.write(f"### 🏆 ケリー基準推奨馬券（予算: {budget:,}円）")
        df_final = df_race[['馬番', '馬名', '複勝オッズ_下限', 'AI複勝率', '複勝期待値(EV)', '推奨金額(円)', 'おすすめ度']].copy()
        df_final['複勝期待値(EV)'] = df_final['複勝期待値(EV)'].map('{:.2f}'.format)
        
        # 推奨金額が0円の馬は下へ、期待値順にソート
        st.dataframe(df_final.sort_values(by=['推奨金額(円)', '複勝期待値(EV)'], ascending=[False, False]), use_container_width=True, hide_index=True)
        
        with st.expander("🕵️ 五大老AI 衆議（個別予測・単勝オッズ）"):
            cols_to_show = ['馬番', '馬名', '単勝オッズ'] + model_cols
            st.dataframe(df_raceお[cols_to_show], use_container_width=True, hide_index=True)
