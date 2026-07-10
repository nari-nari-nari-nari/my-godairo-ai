import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import re
import requests
from bs4 import BeautifulSoup
import time

# --- ページ設定（一番最初に行う必要があります） ---
st.set_page_config(page_title="🔱大黒天AI V7", layout="wide", page_icon="🔱")

# ==========================================
# 🎨 プロ仕様 カスタムCSSの注入 (ダーク＆ゴールドテーマ)
# ==========================================
st.markdown("""
<style>
/* 全体の文字色と背景の微調整 */
.main {
    background-color: #0E1117;
}
/* タイトルを黄金のグラデーションに */
h1 {
    background: -webkit-linear-gradient(45deg, #FFD700, #FFA500, #FF4500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900 !important;
    letter-spacing: 2px;
}
/* サブ見出しの装飾 */
h3 {
    color: #FFD700 !important;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
}
/* プライマリボタン（一撃予想ボタン）の高級化 */
div.stButton > button[data-baseweb="button"] {
    background: linear-gradient(45deg, #FFD700, #FF8C00);
    color: #000000;
    border: none;
    border-radius: 8px;
    font-weight: 900;
    font-size: 18px;
    padding: 10px 24px;
    box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
    transition: all 0.3s ease;
}
/* ボタンにカーソルを合わせた時のエフェクト */
div.stButton > button[data-baseweb="button"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6);
}
/* ボタンを押した時のエフェクト */
div.stButton > button[data-baseweb="button"]:active {
    transform: translateY(2px);
}
/* メトリック（ダッシュボード）の装飾 */
div[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
/* サイドバーの背景色を少しリッチに */
section[data-testid="stSidebar"] {
    background-color: #11151c;
    border-right: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# --- モデルと記憶のロード ---
@st.cache_resource(show_spinner=False)
def load_models_v7():
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

@st.cache_data(show_spinner=False)
def load_mappings_v7():
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

models = load_models_v7()
jockey_map, trainer_map, memory_df = load_mappings_v7()

# --- 超・ガバガバ読み取り機能（手動コピペ用） ---
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

# --- 🎯 ネット競馬から「絶対に」間違えずに条件と馬を抜く関数 ---
def fetch_race_details_and_horses(race_id_str):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id_str}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://race.netkeiba.com/',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8'
    }
    try:
        time.sleep(1.0)
        session = requests.Session()
        res = session.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return None, f"HTTP Error {res.status_code}", "芝", 1600, "晴", "良"
        
        html_text = res.content.decode('euc-jp', errors='ignore')
        soup = BeautifulSoup(html_text, 'html.parser')
        
        course_type, distance, weather, ground, venue_name = "芝", 1600, "晴", "良", "その他"
        
        race_data01 = soup.select_one('.RaceData01')
        if race_data01:
            text01 = race_data01.text.strip()
            if 'ダ' in text01 or 'ダート' in text01: course_type = "ダート"
            elif '障' in text01 or '障害' in text01: course_type = "障害"
            m_dist = re.search(r'(\d+)m', text01)
            if m_dist: distance = int(m_dist.group(1))
            if '天候:曇' in text01: weather = "曇"
            elif '天候:雨' in text01: weather = "雨"
            elif '天候:小雨' in text01: weather = "小雨"
            elif '天候:雪' in text01: weather = "雪"
            if '馬場:稍' in text01: ground = "稍重"
            elif '馬場:重' in text01: ground = "重"
            elif '馬場:不' in text01: ground = "不良"
            
        race_data02 = soup.select_one('.RaceData02')
        if race_data02:
            text02 = race_data02.text.strip()
            for v in ["東京", "中山", "京都", "阪神", "中京", "札幌", "函館", "福島", "新潟", "小倉", "大井", "川崎", "船橋", "浦和"]:
                if v in text02:
                    venue_name = v
                    break
                    
        horses = []
        for tr in soup.select('.HorseList'):
            umaban_td = tr.select_one('td.Umaban')
            horse_td = tr.select_one('td.HorseInfo .HorseName a')
            if umaban_td and horse_td:
                u_str = umaban_td.text.strip()
                name = horse_td.text.strip()
                if u_str.isdigit():
                    horses.append({'馬番': int(u_str), '馬名': name, '枠番':0, '単勝オッズ_仮':10.0})
                    
        return pd.DataFrame(horses), venue_name, course_type, distance, weather, ground
    except Exception as e:
        return None, "その他", "芝", 1600, "晴", "良"

# --- 🎯 オッズ取得API ---
def fetch_real_odds(race_id):
    base_url = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={}&action=init&type={}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    odds_dict = {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}}
    
    # 🌟 NEW: オッズが "***" などの文字列だった場合に安全にスルーする関数
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    session = requests.Session()
    
    # 単勝・複勝 (type=1) - 以前と同じですが安全に取得
    try:
        res1 = session.get(base_url.format(race_id, 1), headers=headers, timeout=5)
        data1 = res1.json().get('data', {}).get('odds', {})
        if '1' in data1: # 単勝
            for k, v in data1['1'].items():
                odds_dict['win'][int(k)] = safe_float(v[0])
        if '2' in data1: # 複勝
            for k, v in data1['2'].items():
                odds_dict['place'][int(k)] = safe_float(v[0])
    except Exception as e:
        print(f"Win/Place Odds Error: {e}")
        
    # 馬連 (type=4) - データ構造の変更に対応
    try:
        res4 = session.get(base_url.format(race_id, 4), headers=headers, timeout=5)
        data4 = res4.json().get('data', {}).get('odds', {})
        for k1, v1_dict in data4.items():
            if isinstance(v1_dict, dict): # 念のため辞書型か確認
                for k2, v2 in v1_dict.items():
                    u1, u2 = sorted([int(k1), int(k2)])
                    # v2はリストで最初の要素がオッズ
                    if isinstance(v2, list) and len(v2) > 0:
                        odds_dict['quinella'][f"{u1}-{u2}"] = safe_float(v2[0])
    except Exception as e:
         print(f"Quinella Odds Error: {e}")
        
    # ワイド (type=5) - データ構造の変更に対応
    try:
        res5 = session.get(base_url.format(race_id, 5), headers=headers, timeout=5)
        data5 = res5.json().get('data', {}).get('odds', {})
        for k1, v1_dict in data5.items():
             if isinstance(v1_dict, dict):
                for k2, v2 in v1_dict.items():
                    u1, u2 = sorted([int(k1), int(k2)])
                    if isinstance(v2, list) and len(v2) > 0:
                         odds_dict['wide'][f"{u1}-{u2}"] = safe_float(v2[0])
    except Exception as e:
        print(f"Wide Odds Error: {e}")
        
    return odds_dict

# --- 🧠 Harvilleモデル（確率計算） ---
def calculate_exact_multi_probabilities(win_probs):
    n = len(win_probs)
    eps = 1e-9
    win_probs = np.array(win_probs) + eps
    win_probs /= win_probs.sum()
    place, quinella, wide = np.zeros(n), np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        p1 = win_probs[i]
        for j in range(n):
            if i == j: continue
            p2 = win_probs[j] / (1.0 - p1)
            quinella[i, j] += p1 * p2
            for k in range(n):
                if k == i or k == j: continue
                denom = 1.0 - p1 - win_probs[j]
                p3 = win_probs[k] / denom if denom > 0 else 0
                p123 = p1 * p2 * p3
                place[i] += p123; place[j] += p123; place[k] += p123
                wide[i, j] += p123; wide[j, k] += p123; wide[i, k] += p123
    return place, quinella, wide / 2.0


# ==========================================
# UI 画面構築（サイドバー）
# ==========================================
st.sidebar.markdown("### 🎯🚀 超・一撃予想モード")
st.sidebar.caption("URLを貼るだけで、出馬表・距離・天候・馬場・開催場を裏側で自動ハッキングして予想します！")

url_input = st.sidebar.text_input("① レースのURL (★手動モード時もオッズ取得に必須！)", "")
budget = st.sidebar.number_input("② 今回の軍資金 (円)", value=10000, step=1000)

auto_predict_btn = st.sidebar.button("🚀 URLから全自動で予想を実行！", type="primary")

st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ 手動補正モード（自動取得に失敗した時用）"):
    pasted_text = st.text_area("出馬表をコピペ", height=100)
    manual_venue = st.selectbox("開催場", ["小倉", "東京", "中山", "京都", "阪神", "中京", "札幌", "函館", "福島", "新潟", "大井", "川崎", "船橋", "浦和", "その他"])
    manual_course = st.selectbox("コース", ["ダート", "芝", "障害"])
    manual_dist = st.number_input("距離(m)", value=1000, step=100)
    manual_weather = st.selectbox("天候", ["晴", "曇", "小雨", "雨", "雪"])
    manual_ground = st.selectbox("馬場", ["良", "稍重", "重", "不良"])
    manual_btn = st.button("🛠️ 手動データで予想を実行")

# タイトル
st.title("🔱 大黒天AI V7 - 全券種ケリー搭載")
st.caption("AIによる完全システマチック競馬投資ダッシュボード")

# 実行トリガーの判定
is_run = auto_predict_btn or manual_btn

if is_run:
    if not models:
        st.error("モデルファイルが見つかりません。")
        st.stop()
        
    m = re.search(r'(\d{12})', url_input)
    race_id = m.group(1) if m else re.sub(r'\D', '', url_input)
    
    race_num_str = "??"
    if race_id and len(race_id) >= 2 and race_id[-2:].isdigit():
        race_num_str = str(int(race_id[-2:]))
        
    with st.spinner(f"🏇 レースデータを抽出＆オッズ計算中..."):
        
        if auto_predict_btn:
            if not race_id or len(race_id) < 10:
                st.error("❌ エラー：URL または race_id が正しくありません。")
                st.stop()
            
            df_race, venue, course_type, distance, weather, ground = fetch_race_details_and_horses(race_id)
            if df_race is None or df_race.empty:
                st.error(f"❌ エラー：出馬表の自動取得に失敗しました。ブロックされた可能性があります。手動補正モードをお試しください。")
                st.stop()
        else:
            df_race = parse_pasted_text(pasted_text)
            if df_race is None or df_race.empty:
                st.error("❌ 抽出エラー：もう一度出馬表をコピー＆ペーストしてください。")
                st.stop()
            venue, course_type, distance, weather, ground = manual_venue, manual_course, manual_dist, manual_weather, manual_ground
            
            if not race_id or len(race_id) < 10:
                st.warning("⚠️ 画面左上の「① レースのURL」が入力されていないため、最新オッズの取得ができませんでした。\n\nオッズと期待値(EV)は「---」になりますが、AIによる【当たる確率（勝率・複勝率）】の予測ランキングはご覧いただけます。")
            else:
                st.info("🛠️ 手動入力データと最新オッズを組み合わせて予想を実行しました。")

        # 🌟 カッコいいダッシュボード表示
        st.markdown(f"### 📍 【{venue} {race_num_str}R】 レース情報")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏇 条件", f"{course_type} {distance}m")
        col2.metric("🌤 天候", weather)
        col3.metric("🌱 馬場", ground)
        col4.metric("🐎 頭数", f"{len(df_race)}頭立て")
        st.markdown("---")

        # オッズ取得
        real_odds = fetch_real_odds(race_id) if race_id else {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}}
        df_race['単勝オッズ'] = df_race['馬番'].map(lambda x: real_odds['win'].get(x, 10.0))
        df_race['複勝オッズ_下限'] = df_race['馬番'].map(lambda x: real_odds['place'].get(x, 1.1))
        
        venue_dict = {"東京": 0, "中山": 1, "京都": 2, "阪神": 3, "中京": 4, "札幌": 5, "函館": 6, "福島": 7, "新潟": 8, "小倉": 9, "大井": 10, "川崎": 11, "船橋": 12, "浦和": 13, "その他": 14}
        course_num = {"芝": 0, "ダート": 1, "障害": 2}[course_type]
        weather_num = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}[weather]
        ground_num = {"良": 0, "稍重": 1, "重": 2, "不良": 3}[ground]
        
        df_pred = df_race.copy()
        df_pred['コース_num'] = course_num
        df_pred['開催場_num'] = venue_dict.get(venue, 14)
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
            '枠番', '単勝オッズ', '斤量_num', 'コース_num', '開催場_num', '距離', '天候_num', 
            '体重', '年齢', '馬場状態_num', '騎手_num', '調教師_num', '出走間隔_days',
            '全体過去平均着順', '前走着順', '出走回数', '前走脚質_num', '前走上がり3F', '過去平均上がり3F'
        ]
        X_pred = df_pred[features].fillna(0)
        
        preds_dict = {}
        for name, model in models.items():
            preds_dict[name] = model.predict(X_pred)
            
        pred_win_raw = (
            preds_dict['徳川家康予測'] * 0.35 + 
            preds_dict['前田利家予測'] * 0.25 + 
            preds_dict['上杉景勝予測'] * 0.20 + 
            preds_dict['毛利輝元予測'] * 0.12 + 
            preds_dict['宇喜多秀家予測'] * 0.08
        )
        ai_probs = pred_win_raw / (pred_win_raw.sum() if pred_win_raw.sum() > 0 else 1)
        df_race['AI勝率'] = ai_probs
        
        ai_place, ai_quinella, ai_wide = calculate_exact_multi_probabilities(ai_probs)
        df_race['AI複勝率_num'] = ai_place
        
        df_race['複勝期待値(EV)'] = df_race['AI複勝率_num'] * df_race['複勝オッズ_下限']
        df_race['b_value'] = df_race['複勝オッズ_下限'] - 1.0
        df_race['ケリー割合'] = np.where(
            df_race['b_value'] > 0, 
            (df_race['AI複勝率_num'] * df_race['b_value'] - (1.0 - df_race['AI複勝率_num'])) / df_race['b_value'], 0)
        df_race['ケリー割合'] = np.clip(df_race['ケリー割合'], 0, 1)
        df_race['推奨金額(円)'] = (budget * df_race['ケリー割合'] * 0.5).astype(int)
        df_race['推奨金額(円)'] = (df_race['推奨金額(円)'] // 100) * 100
        
        def get_rank(ev, amt, odds):
            if odds == 0 or odds == 10.0: return "---"
            if ev >= 1.20 and amt > 0: return "👑 勝負"
            elif ev >= 1.05 and amt > 0: return "🔥 狙い"
            return "見送り"
            
        df_race['おすすめ度'] = df_race.apply(lambda row: get_rank(row['複勝期待値(EV)'], row['推奨金額(円)'], row['複勝オッズ_下限']), axis=1)
        
        pair_bets = []
        all_pair_stats = [] 
        n = len(df_race)
        horses, horse_names = df_race['馬番'].values, df_race['馬名'].values
        r_quinella, r_wide = real_odds.get('quinella', {}), real_odds.get('wide', {})
        
        for i in range(n):
            for j in range(i+1, n):
                u1, u2 = int(horses[i]), int(horses[j])
                name1, name2 = horse_names[i], horse_names[j]
                pair_key, pair_name = f"{min(u1, u2)}-{max(u1, u2)}", f"{name1} × {name2}"
                
                # 馬連
                q_odds = r_quinella.get(pair_key, 0)
                q_prob = ai_quinella[i, j] + ai_quinella[j, i]
                q_ev = q_prob * q_odds if q_odds > 0 else 0.0
                all_pair_stats.append({'券種': '馬連', '馬番': pair_key, '組み合わせ': pair_name, '確率': q_prob, 'オッズ': q_odds, '期待値(EV)': q_ev})
                if q_odds > 0 and q_ev >= 1.15:
                    b_val = q_odds - 1.0
                    k_frac = max(0, (q_prob * b_val - (1 - q_prob)) / b_val)
                    k_amt = (int(budget * min(k_frac, 1.0) * 0.25) // 100) * 100
                    if k_amt > 0:
                        pair_bets.append({'券種': '馬連', '馬番': pair_key, '組み合わせ': pair_name, '確率': q_prob, 'オッズ': q_odds, '期待値(EV)': q_ev, '推奨金額(円)': k_amt})
                
                # ワイド
                w_odds = r_wide.get(pair_key, 0)
                w_prob = ai_wide[i, j]
                w_ev = w_prob * w_odds if w_odds > 0 else 0.0
                all_pair_stats.append({'券種': 'ワイド', '馬番': pair_key, '組み合わせ': pair_name, '確率': w_prob, 'オッズ': w_odds, '期待値(EV)': w_ev})
                if w_odds > 0 and w_ev >= 1.15:
                    b_val = w_odds - 1.0
                    k_frac = max(0, (w_prob * b_val - (1 - w_prob)) / b_val)
                    k_amt = (int(budget * min(k_frac, 1.0) * 0.25) // 100) * 100
                    if k_amt > 0:
                        pair_bets.append({'券種': 'ワイド', '馬番': pair_key, '組み合わせ': pair_name, '確率': w_prob, 'オッズ': w_odds, '期待値(EV)': w_ev, '推奨金額(円)': k_amt})
                            
        df_pairs = pd.DataFrame(pair_bets)
        df_all_pairs = pd.DataFrame(all_pair_stats)

        # ----------------------------------------------------
        # 画面表示
        # ----------------------------------------------------
        st.write("### 🏆 複勝 期待値＆ケリー推奨")
        df_race['AI複勝率'] = (df_race['AI複勝率_num'] * 100).map('{:.1f}%'.format)
        
        # 色付け関数
        def style_recommendation(val):
            if val == '👑 勝負': return 'color: #FFD700; font-weight: bold;'
            elif val == '🔥 狙い': return 'color: #FF8C00; font-weight: bold;'
            return ''
            
        df_final = df_race[['馬番', '馬名', '複勝オッズ_下限', 'AI複勝率', '複勝期待値(EV)', '推奨金額(円)', 'おすすめ度']].copy()
        df_final['複勝期待値(EV)'] = df_final.apply(lambda row: '{:.2f}'.format(row['複勝期待値(EV)']) if row['複勝オッズ_下限'] != 0 and row['複勝オッズ_下限'] != 10.0 else "---", axis=1)
        df_final['複勝オッズ_下限'] = df_final['複勝オッズ_下限'].apply(lambda x: x if x != 0 and x != 10.0 else "---")
        
        st.dataframe(df_final.sort_values(by=['推奨金額(円)', 'AI複勝率'], ascending=[False, False]).style.map(style_recommendation, subset=['おすすめ度']), use_container_width=True, hide_index=True)
        
        st.write("---")
        
        if not df_pairs.empty:
            st.write("### 🔗 馬連・ワイド 激アツ推奨買い目")
            df_pairs['期待値(EV)'] = df_pairs['期待値(EV)'].map('{:.2f}'.format)
            df_pairs['確率'] = (df_pairs['確率'] * 100).map('{:.1f}%'.format)
            df_pairs = df_pairs.sort_values(by=['券種', '推奨金額(円)', '期待値(EV)'], ascending=[False, False, False])
            st.dataframe(df_pairs, use_container_width=True, hide_index=True)
        else:
            if not race_id or not real_odds['wide']:
                st.warning(f"⚠️ オッズが発表されていないため、馬連・ワイドの期待値と買い目は計算できませんでした。")
            else:
                st.info(f"ℹ️ 現在のオッズでは、馬連・ワイドで期待値(EV)が1.15を超え、購入対象となる組み合わせはありませんでした。")

        with st.expander("📊 すべての馬連・ワイドの確率・期待値一覧を見る"):
            if not df_all_pairs.empty:
                df_all_pairs['期待値(EV)'] = df_all_pairs.apply(lambda row: '{:.2f}'.format(row['期待値(EV)']) if row['オッズ'] > 0 else "---", axis=1)
                df_all_pairs['オッズ'] = df_all_pairs['オッズ'].apply(lambda x: x if x > 0 else "---")
                df_all_pairs['確率'] = (df_all_pairs['確率'] * 100).map('{:.1f}%'.format)
                df_all_pairs = df_all_pairs.sort_values(by=['券種', '確率'], ascending=[False, False])
                st.dataframe(df_all_pairs, use_container_width=True, hide_index=True)
