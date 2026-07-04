import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import requests
from bs4 import BeautifulSoup
import re

# --- ページ設定 ---
st.set_page_config(page_title="🔱大黒天AI V6", layout="wide")
st.title("🔱 大黒天AI V6 - 爆速ハイブリッド実戦版")

@st.cache_resource
def load_models():
    model_files = {
        "徳川家康予測": "model_ieyasu.txt",
        "前田利家予測": "model_toshiie.txt",
        "上杉景勝予測": "model_kagekatsu.txt",
        "毛利輝元予測": "model_terumoto.txt",
        "宇喜多秀家予測": "model_hideie.txt"
    }
    loaded = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded[name] = lgb.Booster(model_file=path)
    return loaded

@st.cache_data
def load_mappings():
    j_map, t_map = {}, {}
    mem_df = pd.DataFrame()
    try:
        if os.path.exists("jockey_map.json"):
            with open("jockey_map.json", "r", encoding="utf-8") as f: j_map = json.load(f)
        if os.path.exists("trainer_map.json"):
            with open("trainer_map.json", "r", encoding="utf-8") as f: t_map = json.load(f)
        if os.path.exists("lite_memory_df.csv"):
            temp_df = pd.read_csv("lite_memory_df.csv")
            if '馬名' in temp_df.columns:
                mem_df = temp_df.set_index('馬名')
    except Exception:
        pass
    return j_map, t_map, mem_df

models = load_models()
jockey_map, trainer_map, memory_df = load_mappings()

# --- 極限まで人間になりすますステルス・スクレイピング ---
@st.cache_data(ttl=300)
def fetch_race_data(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    
    # 🕵️ 人間のブラウザが送る「細かい指紋」を完全コピー
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        'Referer': 'https://race.netkeiba.com/',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'Connection': 'keep-alive'
    }
    
    try:
        # Sessionを使って「さっきからサイトを見てる普通の人間感」を演出
        session = requests.Session()
        # トップページに一瞬アクセスしてCookieをもらう（偵察）
        session.get("https://race.netkeiba.com/", headers=headers, timeout=5)
        
        # 本命のページにアクセス
        res = session.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        res.encoding = 'EUC-JP'
        soup = BeautifulSoup(res.text, 'html.parser')
        
        course_type = "芝"
        distance = 1600
        race_data_elem = soup.select_one('.RaceData01')
        if race_data_elem:
            text = race_data_elem.text
            if "ダ" in text: course_type = "ダート"
            elif "障" in text: course_type = "障害"
            
            m = re.search(r'(\d+)m', text)
            if m: distance = int(m.group(1))
            
        horses_data = []
        for row in soup.select('.HorseList'):
            try:
                umaban = int(row.select_one('.Umaban').text.strip())
                wakuban = int(row.select_one('.Waku').text.strip() or 0)
                horse_name = row.select_one('.HorseName').text.strip()
                jockey = row.select_one('.Jockey').text.strip()
                trainer = row.select_one('.Trainer').text.strip()
                
                odds_elem = row.select_one('.Txt_C')
                odds = 10.0
                if odds_elem and odds_elem.text.strip().replace('.','',1).isdigit():
                    odds = float(odds_elem.text.strip())
                
                horses_data.append({
                    '馬番': umaban, '枠番': wakuban, '馬名': horse_name,
                    '騎手': jockey, '調教師': trainer, '単勝オッズ': odds
                })
            except:
                continue
                
        if not horses_data:
            return None, "芝", 1600, "馬のデータが空です（HTML構造の変更かブロックの可能性）"
            
        return pd.DataFrame(horses_data), course_type, distance, "成功"
        
    except Exception as e:
        return None, "芝", 1600, f"侵入失敗: {str(e)}"

# --- サイドバー：予想設定 ---
st.sidebar.header("🎯 実戦レース設定")
st.sidebar.markdown("ネット競馬のURLをそのまま貼ってください。")
url_input = st.sidebar.text_input("ネット競馬の出馬表URL", "")
st.sidebar.markdown("---")
st.sidebar.markdown("🌤️ 当日の状況（手動）")
weather = st.sidebar.selectbox("天候", ["晴", "曇", "小雨", "雨", "雪"])
ground = st.sidebar.selectbox("馬場", ["良", "稍重", "重", "不良"])

weather_num = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}[weather]
ground_num = {"良": 0, "稍重": 1, "重": 2, "不良": 3}[ground]

if st.sidebar.button("🔱 予想を実行"):
    if not models:
        st.error("モデルファイルが見つかりません。")
        st.stop()
        
    if not url_input:
        st.warning("⚠️ URLを入力してください。")
        st.stop()
        
    m = re.search(r'race_id=(\d+)', url_input)
    race_id = m.group(1) if m else url_input.strip()
        
    with st.spinner("🏇 ネット競馬の防壁をハッキング中..."):
        df_race, fetched_course, fetched_dist, error_msg = fetch_race_data(race_id)
        
    if df_race is None or df_race.empty:
        st.error(f"❌ 侵入を検知されました。")
        st.error(f"詳細: {error_msg}")
    else:
        st.success(f"✅ 防壁突破！{len(df_race)}頭のデータと（{fetched_course} {fetched_dist}m）を奪取しました。")
        
        course_num = {"芝": 0, "ダート": 1, "障害": 2}.get(fetched_course, 0)
        
        df_pred = df_race.copy()
        df_pred['コース_num'] = course_num
        df_pred['距離'] = fetched_dist
        df_pred['天候_num'] = weather_num
        df_pred['馬場状態_num'] = ground_num
        df_pred['斤量_num'] = 55.0
        df_pred['体重'] = 480.0
        df_pred['年齢'] = 3.0
        df_pred['出走間隔_days'] = 60.0
        
        df_pred['騎手_num'] = df_pred['騎手'].map(jockey_map).fillna(-1) if jockey_map else -1
        df_pred['調教師_num'] = df_pred['調教師'].map(trainer_map).fillna(-1) if trainer_map else -1
        
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
        df_race['AI複勝率'] = df_race[model_cols].mean(axis=1)
        
        for col in model_cols + ['AI複勝率']:
            df_race[col] = (df_race[col] * 100).map('{:.1f}%'.format)
            
        st.write(f"### 🏆 最終推奨順位（{fetched_course} {fetched_dist}m / 天候:{weather} / 馬場:{ground}）")
        df_final = df_race[['馬番', '馬名', '単勝オッズ', 'AI複勝率']].copy()
        df_final['ソート用'] = df_final['AI複勝率'].str.replace('%', '').astype(float)
        st.dataframe(df_final.sort_values(by='ソート用', ascending=False).drop(columns=['ソート用']), use_container_width=True, hide_index=True)