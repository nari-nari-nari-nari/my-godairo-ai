import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import re
import requests

# --- ページ設定 ---
st.set_page_config(page_title="🔱大黒天AI V7", layout="wide")
st.title("🔱 大黒天AI V7 - 全券種（馬連・ワイド）ケリー基準搭載版")

# --- 軽量モデルと過去の記憶のロード (v7としてキャッシュを強制リセット) ---
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

# --- 🎯 馬連・ワイド対応 最新オッズ取得API ---
def fetch_real_odds(race_id):
    base_url = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={}&action=init&type={}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    odds_dict = {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}}
    try:
        res1 = requests.get(base_url.format(race_id, 1), headers=headers, timeout=5)
        d1 = res1.json().get('data', {}).get('odds', {})
        if '1' in d1:
            for k, v in d1['1'].items(): odds_dict['win'][int(k)] = float(v[0])
        if '2' in d1:
            for k, v in d1['2'].items(): odds_dict['place'][int(k)] = float(v[0])
            
        res4 = requests.get(base_url.format(race_id, 4), headers=headers, timeout=5)
        d4 = res4.json().get('data', {}).get('odds', {})
        for k1, v1_dict in d4.items():
            for k2, v2 in v1_dict.items():
                u1, u2 = sorted([int(k1), int(k2)])
                odds_dict['quinella'][f"{u1}-{u2}"] = float(v2[0])
                
        res5 = requests.get(base_url.format(race_id, 5), headers=headers, timeout=5)
        d5 = res5.json().get('data', {}).get('odds', {})
        for k1, v1_dict in d5.items():
            for k2, v2 in v1_dict.items():
                u1, u2 = sorted([int(k1), int(k2)])
                odds_dict['wide'][f"{u1}-{u2}"] = float(v2[0])
    except Exception:
        pass
    return odds_dict

# --- 🧠 全組み合わせの確率を計算する数学モデル（Harvilleモデル） ---
def calculate_exact_multi_probabilities(win_probs):
    n = len(win_probs)
    eps = 1e-9
    win_probs = np.array(win_probs) + eps
    win_probs /= win_probs.sum()
    place = np.zeros(n)
    quinella = np.zeros((n, n))
    wide = np.zeros((n, n))
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
                place[i] += p123
                place[j] += p123
                place[k] += p123
                wide[i, j] += p123
                wide[j, k] += p123
                wide[i, k] += p123
    return place, quinella, wide / 2.0

# --- サイドバー：予想設定 ---
st.sidebar.header("🎯 実戦レース設定")
url_input = st.sidebar.text_input("① レースのURL (オッズ取得用)", "")
st.sidebar.markdown("② 出馬表を「適当に全部コピー」してペースト")
pasted_text = st.sidebar.text_area("", height=150)

st.sidebar.markdown("---")
budget = st.sidebar.number_input("💸 今回の軍資金 (円)", value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.markdown("🌤️ レース条件（手動入力）")
venue = st.sidebar.selectbox("開催場", ["東京", "中山", "京都", "阪神", "中京", "札幌", "函館", "福島", "新潟", "小倉", "大井", "川崎", "船橋", "浦和", "その他"])
course_type = st.sidebar.selectbox("コース", ["芝", "ダート", "障害"])
distance = st.sidebar.number_input("距離(m)", value=1600, step=100)
weather = st.sidebar.selectbox("天候", ["晴", "曇", "小雨", "雨", "雪"])
ground = st.sidebar.selectbox("馬場", ["良", "稍重", "重", "不良"])

venue_dict = {"東京": 0, "中山": 1, "京都": 2, "阪神": 3, "中京": 4, "札幌": 5, "函館": 6, "福島": 7, "新潟": 8, "小倉": 9, "大井": 10, "川崎": 11, "船橋": 12, "浦和": 13, "その他": 14}
venue_num = venue_dict[venue]
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
    
    # URLのrace_id（末尾2桁）からレース番号を抽出
    race_num_str = "??"
    if race_id and len(race_id) >= 2 and race_id[-2:].isdigit():
        race_num_str = str(int(race_id[-2:]))
        
    with st.spinner(f"🏇 {venue} {race_num_str}R のデータを抽出＆オッズ計算中..."):
        df_race = parse_pasted_text(pasted_text)
        real_odds = fetch_real_odds(race_id) if race_id else {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}}
        
    if df_race is None or df_race.empty:
        st.error("❌ 抽出不可能なレベルのテキストです。もう一度コピペしてみてください。")
    else:
        df_race['単勝オッズ'] = df_race['馬番'].map(lambda x: real_odds['win'].get(x, 10.0))
        df_race['複勝オッズ_下限'] = df_race['馬番'].map(lambda x: real_odds['place'].get(x, 1.1))
        
        st.success(f"✅ 【抽出成功】{len(df_race)}頭の馬名を発見し、最新オッズを結合しました！")
        
        df_pred = df_race.copy()
        df_pred['コース_num'] = course_num
        df_pred['開催場_num'] = venue_num
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
        
        # 19個の特徴量（V7）
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
        
        pair_bets = []
        n = len(df_race)
        horses = df_race['馬番'].values
        horse_names = df_race['馬名'].values
        r_quinella = real_odds.get('quinella', {})
        r_wide = real_odds.get('wide', {})
        
        for i in range(n):
            for j in range(i+1, n):
                u1, u2 = int(horses[i]), int(horses[j])
                name1, name2 = horse_names[i], horse_names[j]
                
                pair_key = f"{min(u1, u2)}-{max(u1, u2)}"
                pair_name = f"{name1} × {name2}"
                
                # 馬連
                q_odds = r_quinella.get(pair_key, 0)
                if q_odds > 0:
                    q_prob = ai_quinella[i, j] + ai_quinella[j, i]
                    q_ev = q_prob * q_odds
                    if q_ev >= 1.15:
                        b_val = q_odds - 1.0
                        k_frac = max(0, (q_prob * b_val - (1 - q_prob)) / b_val)
                        k_amt = int(budget * min(k_frac, 1.0) * 0.25)
                        k_amt = (k_amt // 100) * 100
                        if k_amt > 0:
                            pair_bets.append({'券種': '馬連', '馬番': pair_key, '組み合わせ': pair_name, '確率': q_prob, 'オッズ': q_odds, '期待値(EV)': q_ev, '推奨金額(円)': k_amt})
                
                # ワイド
                w_odds = r_wide.get(pair_key, 0)
                if w_odds > 0:
                    w_prob = ai_wide[i, j]
                    w_ev = w_prob * w_odds
                    if w_ev >= 1.15:
                        b_val = w_odds - 1.0
                        k_frac = max(0, (w_prob * b_val - (1 - w_prob)) / b_val)
                        k_amt = int(budget * min(k_frac, 1.0) * 0.25)
                        k_amt = (k_amt // 100) * 100
                        if k_amt > 0:
                            pair_bets.append({'券種': 'ワイド', '馬番': pair_key, '組み合わせ': pair_name, '確率': w_prob, 'オッズ': w_odds, '期待値(EV)': w_ev, '推奨金額(円)': k_amt})
                            
        df_pairs = pd.DataFrame(pair_bets)

        # ----------------------------------------------------
        # 画面表示
        # ----------------------------------------------------
        st.write(f"### 🏁 【{venue} {race_num_str}R】 🏆 複勝 期待値＆ケリー推奨")
        
        df_race['AI複勝率'] = (df_race['AI複勝率_num'] * 100).map('{:.1f}%'.format)
        model_cols = ['徳川家康予測', '前田利家予測', '上杉景勝予測', '毛利輝元予測', '宇喜多秀家予測']
        for name in model_cols:
            win_prob = preds_dict[name] / (preds_dict[name].sum() if preds_dict[name].sum() > 0 else 1)
            df_race[name] = (np.clip(1 - (1 - win_prob) ** 2.85, 0, 1) * 100).map('{:.1f}%'.format)
            
        df_final = df_race[['馬番', '馬名', '複勝オッズ_下限', 'AI複勝率', '複勝期待値(EV)', '推奨金額(円)', 'おすすめ度']].copy()
        df_final['複勝期待値(EV)'] = df_final['複勝期待値(EV)'].map('{:.2f}'.format)
        st.dataframe(df_final.sort_values(by=['推奨金額(円)', '複勝期待値(EV)'], ascending=[False, False]), use_container_width=True, hide_index=True)
        
        st.write("---")
        
        if not df_pairs.empty:
            st.write(f"### 🔗 【{venue} {race_num_str}R】 馬連・ワイド 激アツ推奨買い目")
            df_pairs['期待値(EV)'] = df_pairs['期待値(EV)'].map('{:.2f}'.format)
            df_pairs['確率'] = (df_pairs['確率'] * 100).map('{:.1f}%'.format)
            
            df_pairs = df_pairs.sort_values(by=['券種', '推奨金額(円)', '期待値(EV)'], ascending=[False, False, False])
            st.dataframe(df_pairs, use_container_width=True, hide_index=True)
        else:
            st.info(f"ℹ️ {venue} {race_num_str}R では、馬連・ワイドで期待値(EV)が1.15を超え、購入対象となる組み合わせはありませんでした。")

        with st.expander("🕵️ 五大老AI 衆議（個別予測・単勝オッズ）"):
            cols_to_show = ['馬番', '馬名', '単勝オッズ'] + model_cols
            st.dataframe(df_race[cols_to_show], use_container_width=True, hide_index=True)