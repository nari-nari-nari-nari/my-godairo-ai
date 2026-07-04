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
st.title("🔱 大黒天AI V6 - 五大老全員集結・完全実戦版")

# --- 軽量モデルと過去の記憶のロード（厳密チェックモード） ---
@st.cache_resource
def load_models_strict():
    model_files = {
        "徳川家康予測": "model_ieyasu.txt",
        "前田利家予測": "model_toshiie.txt",
        "上杉景勝予測": "model_kagekatsu.txt",
        "毛利輝元予測": "model_terumoto.txt",
        "宇喜多秀家予測": "model_hideie.txt"
    }
    loaded = {}
    missing = []
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded[name] = lgb.Booster(model_file=path)
        else:
            missing.append(path)
    return loaded, missing

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

models, missing_models = load_models_strict()
jockey_map, trainer_map, memory_df = load_mappings()

# 👑 5人揃っていない場合は画面上でストップさせる
if missing_models:
    st.error("❌【至急】5人全員で予想するためには、以下のファイルがGitHubに足りません！")
    for f in missing_models:
        st.markdown(f"- **`{f}`**")
    st.info("💡 上記のファイルをGitHubにアップロードすると、この警告が消えて予想が開始されます。")
    st.stop()

# --- 超・ガバガバ読み取り機能 ---
def parse_pasted_text(text):
    horses_data = []
    lines = text.split('\n')
    current_umaban = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if re.match(r'^\d+$', line):
            num = int(line)
            if 1 <= num <= 18:
                current_umaban = num
                continue
                
        m_same = re.match(r'^(\d+)\s+([ァ-ンヴー・]{2,9})', line)
        if m_same:
            umaban = int(m_same.group(1))
            name = m_same.group(2)
            if 1 <= umaban <= 18 and not any(h['馬番'] == umaban for h in horses_data):
                horses_data.append({'馬番': umaban, '馬名': name, '枠番':0, '単勝オッズ_仮':10.0})
            current_umaban = None
            continue
            
        m_name = re.search(r'^[ァ-ンヴー・]{2,9}', line)
        if m_name and current_umaban is not None:
            name = m_name.group(0)
            if not any(h['馬番'] == current_umaban for h in horses_data):
                horses_data.append({'馬番': current_umaban, '馬名': name, '枠番':0, '単勝オッズ_仮':10.0})
            current_umaban = None
            continue

    if not horses_data:
        names = re.findall(r'[ァ-ンヴー・]{2,9}', text)
        ignore_list = ['ルメール', 'デムーロ', 'モレイラ', 'マーカンド', 'ムルザバ', 'レーン', 'ダート', 'コース']
        filtered_names = [n for n in names if n not in ignore_list]
        
        seen = set()
        unique_names = []
        for n in filtered_names:
            if n not in seen:
                unique_names.append(n)
                seen.add(n)
                
        for i, name in enumerate(unique_names):
            if i >= 18: break
            horses_data.append({'馬番': i+1, '馬名': name, '枠番':0, '単勝オッズ_仮':10.0})

    return pd.DataFrame(horses_data) if horses_data else None

# --- ネット競馬の裏APIから「最新オッズ」だけを抜く ---
def fetch_real_odds(race_id):
    base_url = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={}&action=init&type=1"
    headers = {'User-Agent': 'Mozilla/5.0'}
    odds_dict = {'win': {}, 'place': {}}
    try:
        res = requests.get(base_url.format(race_id), headers=headers, timeout=5)
        d = res.json().get('data', {}).get('odds', {})
        if '1' in d:
            for k, v in d['1'].items(): odds_dict['win'][int(k)] = float(v[0])
        if '2' in d:
            for k, v in d['2'].items(): odds_dict['place'][int(k)] = float(v[0])
    except Exception:
        pass
    return odds_dict

# --- サイドバー：予想設定 ---
st.sidebar.header("🎯 実戦レース設定")
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

if st.sidebar.button("🔱 5人全員で予想を実行"):
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
        
        st.success(f"✅ 【五大老 衆議開始】{len(df_race)}頭のデータを読み込み、5人全員での予測を開始します！")
        
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
        
        model_cols = ['徳川家康予測', '前田利家予測', '上杉景勝予測', '毛利輝元予測', '宇喜多秀家予測']
        for name, model in models.items():
            preds = model.predict(X_pred)
            win_prob = preds / (preds.sum() if preds.sum() > 0 else 1)
            df_race[name] = np.clip(1 - (1 - win_prob) ** 2.85, 0, 1)
            
        df_race['AI複勝率_num'] = df_race[model_cols].mean(axis=1)
        df_race['複勝期待値(EV)'] = df_race['AI複勝率_num'] * df_race['複勝オッズ_下限']
        
        df_race['b_value'] = df_race['複勝オッズ_下限'] - 1.0
        df_race['ケリー割合'] = np.where(
            df_race['b_value'] > 0, 
            (df_race['AI複勝率_num'] * df_race['b_value'] - (1.0 - df_race['AI複勝率_num'])) / df_race['b_value'], 
            0
        )
        df_race['ケリー割合'] = np.clip(df_race['ケリー割合'], 0, 1)
        
        df_race['推奨金額(円)'] = (budget * df_race['ケリー割合'] * 0.5).astype(int)
        df_race['推奨金額(円)'] = (df_race['推奨金額(円)'] // 100) * 100
        
        def get_rank(ev, amt):
            if ev >= 1.20 and amt > 0: return "👑 勝負"
            elif ev >= 1.05 and amt > 0: return "🔥 狙い"
            return "見送り"
            
        df_race['おすすめ度'] = df_race.apply(lambda row: get_rank(row['複勝期待値(EV)'], row['推奨金額(円)']), axis=1)
        
        df_race['AI複勝率'] = (df_race['AI複勝率_num'] * 100).map('{:.1f}%'.format)
        for col in model_cols:
            df_race[col] = (df_race[col] * 100).map('{:.1f}%'.format)
            
        st.write(f"### 🏆 5人衆議・お宝馬券ランキング（予算: {budget:,}円）")
        df_final = df_race[['馬番', '馬名', '複勝オッズ_下限', 'AI複勝率', '複勝期待値(EV)', '推奨金額(円)', 'おすすめ度']].copy()
        df_final['複勝期待値(EV)'] = df_final['複勝期待値(EV)'].map('{:.2f}'.format)
        
        st.dataframe(df_final.sort_values(by=['推奨金額(円)', '複勝期待値(EV)'], ascending=[False, False]), use_container_width=True, hide_index=True)
        
        with st.expander("🕵️ 五大老AI 衆議詳細（個別予測・単勝オッズ）"):
            cols_to_show = ['馬番', '馬名', '単勝オッズ'] + model_cols
            st.dataframe(df_race[cols_to_show], use_container_width=True, hide_index=True)

