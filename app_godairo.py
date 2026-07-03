import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import requests
from bs4 import BeautifulSoup

# --- ページ設定 ---
st.set_page_config(page_title="🔱大黒天AI V6", layout="wide")
st.title("🔱 大黒天AI V6 - 爆速ハイブリッド実戦版")

# --- 軽量モデルと過去の記憶のロード ---
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
    if os.path.exists("jockey_map.json"):
        with open("jockey_map.json", "r", encoding="utf-8") as f: j_map = json.load(f)
    if os.path.exists("trainer_map.json"):
        with open("trainer_map.json", "r", encoding="utf-8") as f: t_map = json.load(f)
    if os.path.exists("lite_memory_df.csv"):
        mem_df = pd.read_csv("lite_memory_df.csv", index_col='馬名')
    return j_map, t_map, mem_df

models = load_models()
jockey_map, trainer_map, memory_df = load_mappings()

# --- 本物の出馬表をスクレイピング ---
@st.cache_data(ttl=300)
def fetch_race_data(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'EUC-JP'
        soup = BeautifulSoup(res.text, 'html.parser')
        
        horses_data = []
        rows = soup.select('.HorseList')
        for row in rows:
            try:
                umaban = int(row.select_one('.Umaban').text.strip())
                wakuban = int(row.select_one('.Waku').text.strip() or 0)
                horse_name = row.select_one('.HorseName').text.strip()
                jockey = row.select_one('.Jockey').text.strip()
                trainer = row.select_one('.Trainer').text.strip()
                
                # オッズ取得
                odds_elem = row.select_one('.Txt_C')
                odds = 10.0
                if odds_elem and odds_elem.text.strip().replace('.','',1).isdigit():
                    odds = float(odds_elem.text.strip())
                
                horses_data.append({
                    '馬番': umaban,
                    '枠番': wakuban,
                    '馬名': horse_name,
                    '騎手': jockey,
                    '調教師': trainer,
                    '単勝オッズ': odds
                })
            except:
                continue
        return pd.DataFrame(horses_data) if horses_data else None
    except:
        return None

# --- サイドバー：予想設定 ---
st.sidebar.header("🎯 実戦レース設定")
race_id_input = st.sidebar.text_input("レースID (例: 202606010101)", "202606010101")
course_type = st.sidebar.selectbox("コース", ["芝", "ダート", "障害"])
distance = st.sidebar.number_input("距離(m)", value=1600, step=100)
weather = st.sidebar.selectbox("天候", ["晴", "曇", "小雨", "雨", "雪"])
ground = st.sidebar.selectbox("馬場", ["良", "稍重", "重", "不良"])

# コースや天候をAI用の数値に変換
course_num = {"芝": 0, "ダート": 1, "障害": 2}[course_type]
weather_num = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}[weather]
ground_num = {"良": 0, "稍重": 1, "重": 2, "不良": 3}[ground]

if st.sidebar.button("🔱 予想を実行"):
    if not models:
        st.error("モデルファイルが見つかりません。GitHubにアップロードされているか確認してください。")
        st.stop()
        
    with st.spinner("🏇 ネット競馬から出馬表をハッキング中..."):
        df_race = fetch_race_data(race_id_input)
        
    if df_race is None or df_race.empty:
        st.error("出馬表の取得に失敗しました。レースIDが正しいか確認してください。")
    else:
        st.success(f"✅ {len(df_race)}頭の出走データを取得完了！予測を開始します...")
        
        # 予測用のデータフレーム作成
        df_pred = df_race.copy()
        df_pred['コース_num'] = course_num
        df_pred['距離'] = distance
        df_pred['天候_num'] = weather_num
        df_pred['馬場状態_num'] = ground_num
        
        # デフォルト値補完
        df_pred['斤量_num'] = 55.0
        df_pred['体重'] = 480.0
        df_pred['年齢'] = 3.0
        df_pred['出走間隔_days'] = 60.0
        
        # 騎手・調教師の数値マッピング
        df_pred['騎手_num'] = df_pred['騎手'].map(jockey_map).fillna(-1)
        df_pred['調教師_num'] = df_pred['調教師'].map(trainer_map).fillna(-1)
        
        # メモリファイルから過去の成績を合体
        mem_cols = ['全体過去平均着順', '前走着順', '出走回数', '前走脚質_num', '前走上がり3F', '過去平均上がり3F']
        for col in mem_cols:
            df_pred[col] = df_pred['馬名'].map(lambda x: memory_df.loc[x, col] if x in memory_df.index else np.nan)
        
        # 新馬などの欠損値埋め
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
        
        # 五大老による推論
        for name, model in models.items():
            preds = model.predict(X_pred)
            # ハルヴィル関数で勝率から複勝率へ変換
            win_prob = preds / (preds.sum() if preds.sum() > 0 else 1)
            place_prob = 1 - (1 - win_prob) ** 2.85
            df_race[name] = np.clip(place_prob, 0, 1)
            
        # UI表示用の整形
        model_cols = ['徳川家康予測', '前田利家予測', '上杉景勝予測', '毛利輝元予測', '宇喜多秀家予測']
        df_race['AI複勝率'] = df_race[model_cols].mean(axis=1)
        
        # ％表示に変換
        for col in model_cols + ['AI複勝率']:
            df_race[col] = (df_race[col] * 100).map('{:.1f}%'.format)
            
        st.write("### 🕵️ 五大老AI 衆議（個別予測）")
        cols_to_show = ['馬番', '馬名', '単勝オッズ', '徳川家康予測', '前田利家予測', '上杉景勝予測', '毛利輝元予測', '宇喜多秀家予測']
        st.dataframe(df_race[cols_to_show], use_container_width=True, hide_index=True)
        
        st.write("### 🏆 最終推奨順位")
        df_final = df_race[['馬番', '馬名', '単勝オッズ', 'AI複勝率']].copy()
        # ソート用に%を外して数値に戻して並び替え
        df_final['ソート用'] = df_final['AI複勝率'].str.replace('%', '').astype(float)
        df_final = df_final.sort_values(by='ソート用', ascending=False).drop(columns=['ソート用'])
        st.dataframe(df_final, use_container_width=True, hide_index=True)