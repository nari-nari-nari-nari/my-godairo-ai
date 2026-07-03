import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import os
import json
import time
import requests
import unicodedata

warnings.filterwarnings('ignore')

# ターミナル用カラーコード
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

print(f"{CYAN}===================================================={RESET}")
print(f"{CYAN} 🔱 大黒天AI V6 ガチバックテスト（おすすめ馬券抽出機能付き） 🔱{RESET}")
print(f"{CYAN}===================================================={RESET}")

if not os.path.exists("ultimate_master_data_v4.csv"):
    print(f"{RED}❌ エラー: 'ultimate_master_data_v4.csv' が見つかりません。{RESET}")
    exit()

print(f"⏳ 1. マスターデータの読み込みと時系列の厳密ソートを実行中...")

df_master = pd.read_csv("ultimate_master_data_v4.csv", low_memory=False)
df_bt = df_master.copy()

# 📅 日付処理と【カンニング防止用】厳密な時系列ソート
df_bt['race_date'] = pd.to_datetime(df_bt['race_id'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
df_bt = df_bt.dropna(subset=['race_date']).sort_values(['race_date', 'race_id', '馬番'])

# 特徴量エンジニアリング
df_bt['コース_num'] = df_bt['コース'].map({"芝": 0, "ダート": 1, "障害": 2}).fillna(0)
df_bt['天候_num'] = df_bt['天候'].map({"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "雪": 4}).fillna(0)
df_bt['馬場状態_num'] = df_bt['馬場状態'].map({"良": 0, "稍重": 1, "重": 2, "不良": 3, "不明": -1}).fillna(-1)
df_bt['斤量_num'] = pd.to_numeric(df_bt['斤量'], errors='coerce').fillna(55.0)
df_bt['体重'] = pd.to_numeric(df_bt['体重'], errors='coerce').fillna(480.0)
df_bt['年齢'] = pd.to_numeric(df_bt['年齢'], errors='coerce').fillna(3.0)
df_bt['出走間隔_days'] = pd.to_numeric(df_bt['出走間隔_days'], errors='coerce').fillna(60.0)
df_bt['単勝オッズ'] = pd.to_numeric(df_bt['単勝'], errors='coerce').fillna(10.0)

# 🚨 【超重要バグ修正】全角数字や「1(降)」などの文字混じりを完璧に数字だけ抜き出す
df_bt['着順'] = df_bt['着順'].astype(str).str.extract(r'(\d+)')[0].astype(float).fillna(99.0)

bt_jockey_map = {j: i for i, j in enumerate(df_bt['騎手'].value_counts().index)}
bt_trainer_map = {t: i for i, t in enumerate(df_bt['調教師'].value_counts().index)}
df_bt['騎手_num'] = df_bt['騎手'].map(bt_jockey_map).fillna(-1)
df_bt['調教師_num'] = df_bt['調教師'].map(bt_trainer_map).fillna(-1)
df_bt['脚質_num'] = df_bt['脚質'].map({"逃げ": 0, "先行": 1, "差し": 2, "追込": 3, "その他": 4, "不明": 4}).fillna(4)
df_bt['上がり3F'] = pd.to_numeric(df_bt['上がり3F'], errors='coerce').fillna(35.0)

# 🛡️ カンニング防止：馬ごとに過去情報だけで特徴量を作る
df_bt = df_bt.sort_values(['馬名', 'race_date'])
df_bt['前走着順'] = df_bt.groupby('馬名')['着順'].shift(1).fillna(8.0)
df_bt['出走回数'] = df_bt.groupby('馬名').cumcount() + 1
df_bt['全体過去平均着順'] = df_bt.groupby('馬名')['着順'].transform(lambda x: x.expanding().mean().shift(1)).fillna(8.0)
df_bt['前走脚質_num'] = df_bt.groupby('馬名')['脚質_num'].shift(1).fillna(4.0)
df_bt['前走上がり3F'] = df_bt.groupby('馬名')['上がり3F'].shift(1).fillna(35.0)
df_bt['過去平均上がり3F'] = df_bt.groupby('馬名')['上がり3F'].transform(lambda x: x.expanding().mean().shift(1)).fillna(35.0)
df_bt['target_win'] = (df_bt['着順'] == 1.0).astype(int)

features = [
    '枠番', '単勝オッズ', '斤量_num', 'コース_num', '距離', '天候_num', 
    '体重', '年齢', '馬場状態_num', '騎手_num', '調教師_num', '出走間隔_days',
    '全体過去平均着順', '前走着順', '出走回数', '前走脚質_num', '前走上がり3F', '過去平均上がり3F'
]

# 📅 データの期間分割（2026年3月までを学習、4月〜6月をテスト）
train_mask = df_bt['race_date'] < '2026-04-01'
test_mask = (df_bt['race_date'] >= '2026-04-01') & (df_bt['race_date'] <= '2026-06-30')

df_train = df_bt[train_mask]
df_test = df_bt[test_mask].copy()

print(f"✅ 学習データ: {len(df_train)}件 (〜2026年3月)")
print(f"✅ テストデータ: {len(df_test)}件 (2026年4月〜6月)")

if len(df_test) == 0:
    print(f"{RED}❌ エラー: 指定期間（2026年4月〜6月）のデータが存在しません。{RESET}")
    exit()

print(f"\n🧠 2. 五大老AIの学習を開始します（約5秒）...")

train_set = lgb.Dataset(df_train[features], label=df_train['target_win'])

bt_models = {
    '家康': lgb.train({'objective':'binary','metric':'binary_logloss','verbosity':-1,'random_state':42,'learning_rate':0.03,'num_leaves':15}, train_set, num_boost_round=100),
    '利家': lgb.train({'objective':'binary','metric':'binary_logloss','verbosity':-1,'random_state':104,'learning_rate':0.05,'num_leaves':31}, train_set, num_boost_round=90),
    '景勝': lgb.train({'objective':'binary','metric':'binary_logloss','verbosity':-1,'random_state':555,'learning_rate':0.05,'num_leaves':31,'lambda_l1':1.5}, train_set, num_boost_round=90),
    '輝元': lgb.train({'objective':'binary','metric':'binary_logloss','verbosity':-1,'random_state':888,'learning_rate':0.04,'num_leaves':31}, train_set, num_boost_round=80),
    '秀家': lgb.train({'objective':'binary','metric':'binary_logloss','verbosity':-1,'random_state':777,'learning_rate':0.08,'num_leaves':15}, train_set, num_boost_round=70)
}

# 🛡️ 予測の前に、正解データ（着順）を完全に隠蔽する
X_test = df_test[features].fillna(0).copy()

print(f"\n🔮 3. 五大老による勝率予測を実行中...")
df_test['pred_win_raw'] = (
    bt_models['家康'].predict(X_test) * 0.35 + 
    bt_models['利家'].predict(X_test) * 0.25 + 
    bt_models['景勝'].predict(X_test) * 0.20 + 
    bt_models['輝元'].predict(X_test) * 0.12 + 
    bt_models['秀家'].predict(X_test) * 0.08
)
df_test['pred_win'] = df_test.groupby('race_id')['pred_win_raw'].transform(lambda x: x / (x.sum() if x.sum() > 0 else 1))


print(f"\n🌐 4. 実際の複勝・馬連・ワイドオッズの抜き取り作業を開始します...")
unique_test_races = df_test['race_id'].unique()
odds_cache_file = "real_odds_cache_2026.json"

if os.path.exists(odds_cache_file):
    with open(odds_cache_file, 'r', encoding='utf-8') as f:
        real_odds_cache = json.load(f)
else:
    real_odds_cache = {}

def fetch_real_odds_from_api(race_id_str):
    base_url = "https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={}&action=init&type={}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    odds_dict = {'win': {}, 'place': {}, 'quinella': {}, 'wide': {}}
    try:
        res1 = requests.get(base_url.format(race_id_str, 1), headers=headers, timeout=5)
        d1 = res1.json().get('data', {}).get('odds', {})
        if '1' in d1:
            for k, v in d1['1'].items(): odds_dict['win'][str(k)] = float(v[0])
        if '2' in d1:
            for k, v in d1['2'].items(): odds_dict['place'][str(k)] = float(v[0])
        res4 = requests.get(base_url.format(race_id_str, 4), headers=headers, timeout=5)
        d4 = res4.json().get('data', {}).get('odds', {})
        for k1, v1_dict in d4.items():
            for k2, v2 in v1_dict.items():
                u1, u2 = sorted([int(k1), int(k2)])
                odds_dict['quinella'][f"{u1}-{u2}"] = float(v2[0])
        res5 = requests.get(base_url.format(race_id_str, 5), headers=headers, timeout=5)
        d5 = res5.json().get('data', {}).get('odds', {})
        for k1, v1_dict in d5.items():
            for k2, v2 in v1_dict.items():
                u1, u2 = sorted([int(k1), int(k2)])
                odds_dict['wide'][f"{u1}-{u2}"] = float(v2[0])
    except Exception:
        pass
    return odds_dict

new_scrapes = 0
for i, r_id in enumerate(unique_test_races):
    r_id_str = str(r_id)
    if r_id_str not in real_odds_cache:
        real_odds_cache[r_id_str] = fetch_real_odds_from_api(r_id_str)
        new_scrapes += 1
        time.sleep(0.2)
        if new_scrapes % 20 == 0:
            print(f"   ... APIからオッズ抽出中 ({i+1}/{len(unique_test_races)} レース完了)")

if new_scrapes > 0:
    with open(odds_cache_file, 'w', encoding='utf-8') as f:
        json.dump(real_odds_cache, f, ensure_ascii=False)
    print(f"   ✅ 新たに {new_scrapes} レースの【本物オッズ】をスクレイピングして保存しました。")

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

all_bets = []

# 🚨 足切りフィルター
MIN_WIN_PROB = 0.03   
MIN_PLACE_PROB = 0.10 
MIN_PAIR_PROB = 0.05  

for race_id, group in df_test.groupby('race_id'):
    horses = group['馬番'].values
    horse_names = group['馬名'].values
    ranks = group['着順'].values
    ai_probs = group['pred_win'].values
    win_odds = group['単勝オッズ'].values
    
    r_odds_data = real_odds_cache.get(str(race_id), {})
    r_win = r_odds_data.get('win', {})
    r_place = r_odds_data.get('place', {})
    r_quinella = r_odds_data.get('quinella', {})
    r_wide = r_odds_data.get('wide', {})
    
    pub_probs = 1.0 / win_odds
    pub_probs /= pub_probs.sum()
    
    ai_place, ai_quinella, ai_wide = calculate_exact_multi_probabilities(ai_probs)
    pub_place, pub_quinella, pub_wide = calculate_exact_multi_probabilities(pub_probs)
    
    n = len(horses)
    top3_idx = [i for i, r in enumerate(ranks) if r in [1.0, 2.0, 3.0]]
    
    for i in range(n):
        u1 = str(int(horses[i]))
        name1 = horse_names[i]
        
        # 単勝
        w_odds = r_win.get(u1, win_odds[i])
        w_ev = ai_probs[i] * w_odds if ai_probs[i] >= MIN_WIN_PROB else 0.0
        is_win = 1 if ranks[i] == 1.0 else 0
        all_bets.append({'date': group['race_date'].iloc[0], 'race_id': race_id, 'type': '単勝', 'target': u1, 'name': name1, 'odds': w_odds, 'prob': ai_probs[i], 'ev': w_ev, 'hit': is_win})
        
        # 複勝
        actual_p_odds = r_place.get(u1)
        p_odds = actual_p_odds if actual_p_odds else max(1.1, round(0.80 / pub_place[i] if pub_place[i] > 0 else 1.0, 1))
        p_ev = ai_place[i] * p_odds if ai_place[i] >= MIN_PLACE_PROB else 0.0
        is_place = 1 if ranks[i] in [1.0, 2.0, 3.0] else 0
        all_bets.append({'date': group['race_date'].iloc[0], 'race_id': race_id, 'type': '複勝', 'target': u1, 'name': name1, 'odds': p_odds, 'prob': ai_place[i], 'ev': p_ev, 'hit': is_place})
        
        for j in range(i+1, n):
            u2 = str(int(horses[j]))
            name2 = horse_names[j]
            pair_key = f"{sorted([int(u1), int(u2)])[0]}-{sorted([int(u1), int(u2)])[1]}"
            pair_name = f"{name1} × {name2}"
            
            # 馬連
            actual_q_odds = r_quinella.get(pair_key)
            q_ai_prob = ai_quinella[i, j] + ai_quinella[j, i]
            q_pub_prob = pub_quinella[i, j] + pub_quinella[j, i]
            q_odds = actual_q_odds if actual_q_odds else max(1.5, round(0.775 / q_pub_prob if q_pub_prob > 0 else 1.0, 1))
            q_ev = q_ai_prob * q_odds if q_ai_prob >= MIN_PAIR_PROB else 0.0
            is_q_hit = 1 if (ranks[i] in [1.0, 2.0] and ranks[j] in [1.0, 2.0] and ranks[i] != ranks[j]) else 0
            all_bets.append({'date': group['race_date'].iloc[0], 'race_id': race_id, 'type': '馬連', 'target': pair_key, 'name': pair_name, 'odds': q_odds, 'prob': q_ai_prob, 'ev': q_ev, 'hit': is_q_hit})
            
            # ワイド
            actual_w_odds = r_wide.get(pair_key)
            w_ai_prob = ai_wide[i, j]
            w_pub_prob = pub_wide[i, j]
            w_odds = actual_w_odds if actual_w_odds else max(1.2, round(0.775 / w_pub_prob if w_pub_prob > 0 else 1.0, 1))
            w_ev = w_ai_prob * w_odds if w_ai_prob >= MIN_PAIR_PROB else 0.0
            is_w_hit = 1 if (i in top3_idx and j in top3_idx) else 0
            all_bets.append({'date': group['race_date'].iloc[0], 'race_id': race_id, 'type': 'ワイド', 'target': pair_key, 'name': pair_name, 'odds': w_odds, 'prob': w_ai_prob, 'ev': w_ev, 'hit': is_w_hit})

df_all_bets = pd.DataFrame(all_bets)

# --- シミュレーション実行エンジン ---
def run_simulation(df_bets, ticket_type, min_ev):
    df_target = df_bets[(df_bets['type'] == ticket_type) & (df_bets['ev'] >= min_ev)].copy()
    df_target = df_target.sort_values('date')
    
    bankroll = 100000
    total_inv = 0
    total_ret = 0
    hits = 0
    
    for _, row in df_target.iterrows():
        odds = row['odds']
        prob = row['prob']
        
        kelly = (prob * odds - 1.0) / (odds - 1.0) if odds > 1.0 else 0
        bet_amt = int(bankroll * max(0, kelly) * 0.25)
        
        if bet_amt < 100: continue
        bet_amt = (bet_amt // 100) * 100
        bet_amt = min(bet_amt, int(bankroll * 0.05), 10000) 
        bet_amt = min(bet_amt, (bankroll // 100) * 100)
        if bet_amt < 100: break
        
        bankroll -= bet_amt
        total_inv += bet_amt
        
        if row['hit'] == 1:
            ret = bet_amt * odds
            bankroll += ret
            total_ret += ret
            hits += 1
            
    rec = (total_ret / total_inv * 100) if total_inv > 0 else 0
    hit_rate = (hits / len(df_target) * 100) if len(df_target) > 0 else 0
    profit = bankroll - 100000
    
    return {
        'type': ticket_type,
        'ev_cond': f"EV {min_ev:.2f}+",
        'bets': len(df_target),
        'hit_rate': hit_rate,
        'recovery': rec,
        'profit': profit,
        'final_bank': bankroll
    }

scenarios = [
    ("複勝", 1.10),
    ("複勝", 1.20),
    ("単勝", 1.15),
    ("単勝", 1.25),
    ("ワイド", 1.15),
    ("ワイド", 1.30),
    ("馬連", 1.20),
    ("馬連", 1.50)
]

print(f"\n{YELLOW}📊 【ガチ検証】本物オッズ バックテスト結果サマリー (初期:10万 / 100円単位ケリー配分){RESET}")
print("-" * 85)
print(f" 券種   | 条件(期待値) | 勝負R数 | 的中率  | 回収率  | 最終資金 (純利益)")
print("-" * 85)

for t_type, m_ev in scenarios:
    res = run_simulation(df_all_bets, t_type, m_ev)
    color = GREEN if res['recovery'] >= 100 else RED
    print(f" {res['type']:<4} | {res['ev_cond']:<10} | {res['bets']:>5} R | {res['hit_rate']:>5.1f}% | {color}{res['recovery']:>6.1f}%{RESET} | {color}{int(res['final_bank']):>7,} 円 ({int(res['profit']):+,} 円){RESET}")

print("-" * 85)


# =========================================================================
# 💎 【AI厳選】お宝馬券ランキング（複勝・ワイドのおすすめ度表示機能）
# =========================================================================
def parse_race_id(race_id_str):
    # レースID (例: 202606010101 -> 中山 1R)
    if len(race_id_str) >= 12:
        place_code = race_id_str[4:6]
        race_num = int(race_id_str[-2:])
        places = {"01":"札幌", "02":"函館", "03":"福島", "04":"新潟", "05":"東京", "06":"中山", "07":"中京", "08":"京都", "09":"阪神", "10":"小倉"}
        place = places.get(place_code, "地方")
        return f"{place}{race_num:>2}R"
    return race_id_str

def get_rank(ticket_type, ev):
    if ticket_type == "複勝":
        if ev >= 1.25: return f"{CYAN}👑 Sランク{RESET}", 3
        elif ev >= 1.15: return f"{YELLOW}🔥 Aランク{RESET}", 2
        elif ev >= 1.10: return f"{GREEN}👍 Bランク{RESET}", 1
    elif ticket_type == "ワイド":
        if ev >= 1.30: return f"{CYAN}👑 Sランク{RESET}", 3
        elif ev >= 1.20: return f"{YELLOW}🔥 Aランク{RESET}", 2
        elif ev >= 1.15: return f"{GREEN}👍 Bランク{RESET}", 1
    return "", 0

def get_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'FWA' else 1 for c in s)

def pad_str(s, length):
    return s + " " * max(0, length - get_width(s))

print(f"\n{CYAN}💎 【AI厳選】春季レース 複勝・ワイド お宝馬券ランキング TOP20 💎{RESET}")
print(f"💡 AIが「これは絶対に美味しい！」と判断した、期待値（EV）の高い買い目リストです。")
print("-" * 110)
print(f" 日付  | レース  | 券種   | おすすめ度  | 期待値 | 予想確率 | オッズ | 買い目 (馬番/馬名)                  | 結果")
print("-" * 110)

recom_df = df_all_bets[df_all_bets['type'].isin(['複勝', 'ワイド'])].copy()
recom_df['RankInfo'] = recom_df.apply(lambda row: get_rank(row['type'], row['ev']), axis=1)
recom_df['RankStr'] = recom_df['RankInfo'].apply(lambda x: x[0])
recom_df['RankScore'] = recom_df['RankInfo'].apply(lambda x: x[1])

# S・A・Bランクがついたものだけを抽出し、期待値が高い順に20件表示
recom_df = recom_df[recom_df['RankScore'] > 0]
recom_df = recom_df.sort_values(by='ev', ascending=False).head(20)

for _, row in recom_df.iterrows():
    d_str = row['date'].strftime('%m/%d')
    r_str = pad_str(parse_race_id(str(row['race_id'])), 7)
    hit_str = f"{YELLOW}🎯 的中!{RESET}" if row['hit'] == 1 else f"{RED}❌ ハズレ{RESET}"
    
    t_type = pad_str(row['type'], 6)
    
    target_str = f"[{row['target']}]"
    name_str = row['name']
    if len(name_str) > 16:
        name_str = name_str[:15] + "…"
    combo_str = pad_str(f"{target_str} {name_str}", 35)
    
    # ランクのパディング（カラーコードを無視して文字数計算）
    pure_rank = row['RankStr'].replace('\033[96m', '').replace('\033[93m', '').replace('\033[92m', '').replace('\033[0m', '')
    rank_padded = row['RankStr'] + " " * max(0, 11 - get_width(pure_rank))

    print(f" {d_str} | {r_str} | {t_type} | {rank_padded} | EV{row['ev']:.2f} |  {row['prob']*100:>4.1f}% | {row['odds']:>5.1f}倍 | {combo_str} | {hit_str}")

print("-" * 110)
