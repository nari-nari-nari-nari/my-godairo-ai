import pandas as pd
import numpy as np

print("🧪 データの錬金術（特徴量エンジニアリング）を開始します...")

# 1. データの読み込み
df = pd.read_csv("ultimate_master_data.csv", low_memory=False)

# -----------------------------------
# 🔥 新特徴量①：脚質（展開）の数値化
# -----------------------------------
def calc_kyakushitsu(pass_str):
    if pd.isna(pass_str) or pass_str == '不明' or pass_str == '':
        return 0 # 不明
    try:
        # '2-2-2' -> [2, 2, 2] に分解して平均位置を出す
        ranks = [int(x) for x in str(pass_str).split('-') if x.isdigit()]
        if not ranks: return 0
        avg_rank = sum(ranks) / len(ranks)
        
        # 競馬のセオリーに基づいて分類
        if avg_rank <= 2.5: return 1   # 逃げ
        elif avg_rank <= 6.5: return 2 # 先行
        elif avg_rank <= 10.5: return 3 # 差し
        else: return 4                 # 追込
    except:
        return 0

df['脚質_num'] = df['通過順'].apply(calc_kyakushitsu)
print("✅ 脚質（逃げ・先行・差し・追込）の数値化完了！")

# -----------------------------------
# 🔥 新特徴量②：コース専用実績
# -----------------------------------
# 過去のデータを使うため、時系列（日付順）に正しく並び替える
df = df.sort_values(['馬名', 'race_id']).reset_index(drop=True)

# 🌟 修正ポイント： apply ではなく transform を使うことでインデックスエラーを完全回避！
# A. 全体の過去平均着順
df['全体過去平均着順'] = df.groupby('馬名')['着順'].transform(lambda x: x.shift().expanding().mean())

# B. コース・距離に特化した過去平均着順（これが適性の要！）
df['同コース過去平均着順'] = df.groupby(['馬名', 'コース', '距離'])['着順'].transform(lambda x: x.shift().expanding().mean())

# C. 初挑戦のコースの場合は、全体の過去平均着順で代用し、それもなければ8.0（中間）とする
df['同コース過去平均着順'] = df['同コース過去平均着順'].fillna(df['全体過去平均着順']).fillna(8.0)
df['全体過去平均着順'] = df['全体過去平均着順'].fillna(8.0)

print("✅ コース専用実績の算出完了！")

# 2. 保存
df.to_csv("ultimate_master_data_v2.csv", index=False)
print("📦 完成！ 'ultimate_master_data_v2.csv' に保存しました。")