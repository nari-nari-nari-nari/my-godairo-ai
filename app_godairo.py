import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# ページ設定
st.set_page_config(page_title="🔱大黒天AI V6", layout="wide")
st.title("🔱 大黒天AI V6 - 爆速ハイブリッド版")

@st.cache_resource
def load_models():
    """軽量モデルファイルを一括ロード"""
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

# モデル読み込み
models = load_models()

# 予想実行ボタン
if st.button("🔱 予想を実行"):
    # ダミーデータフレームを作成（実際の運用時はここにデータ処理を繋げてください）
    # ※KeyError回避のため、必ず五大老の名前をカラムとして定義します
    data = {
        "馬番": [1, 2, 3, 4],
        "馬名": ["テスト馬1", "テスト馬2", "テスト馬3", "テスト馬4"],
        "徳川家康予測": [0.1, 0.2, 0.3, 0.4],
        "前田利家予測": [0.1, 0.2, 0.3, 0.4],
        "上杉景勝予測": [0.1, 0.2, 0.3, 0.4],
        "毛利輝元予測": [0.1, 0.2, 0.3, 0.4],
        "宇喜多秀家予測": [0.1, 0.2, 0.3, 0.4]
    }
    df_display = pd.DataFrame(data)

    # 五大老AI 衆議アコーディオン
    with st.expander("🕵️ 五大老AI 衆議（それぞれの個別複勝予測率を見る）"):
        # 必要なカラムのみ抽出して表示
        cols_to_use = ['馬番', '馬名', '徳川家康予測', '前田利家予測', '上杉景勝予測', '毛利輝元予測', '宇喜多秀家予測']
        # カラムが存在するかチェックしてから表示
        existing_cols = [c for c in cols_to_use if c in df_display.columns]
        st.dataframe(df_display[existing_cols], use_container_width=True, hide_index=True)

    # AI複勝率の算出（例）
    model_cols = ['徳川家康予測', '前田利家予測', '上杉景勝予測', '毛利輝元予測', '宇喜多秀家予測']
    df_display['AI複勝率'] = df_display[model_cols].mean(axis=1)
    
    st.write("### 🏆 最終推奨順位")
    st.dataframe(df_display[['馬番', '馬名', 'AI複勝率']].sort_values(by='AI複勝率', ascending=False), use_container_width=True, hide_index=True)

st.info("💡 モデルファイルが正しく配置されているか確認してください。")