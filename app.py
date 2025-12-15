# app.py
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ==================================================
# 基本設定
# ==================================================
st.set_page_config(page_title="糖尿病リスク予測（学習用）", layout="centered")
st.title("糖尿病リスクチェック＆生活習慣改善サポート（学習用）")
st.caption(
    "Pima Indians Diabetes Dataset（オープンデータ）を用いた学習・デモ用アプリです。"
    " 数値の入力から、リスクの把握と生活習慣改善のヒントを提示します。"
)

# ==================================================
# CSS：高さズレを出さない（固定高 + row描画）
# ==================================================
st.markdown(
    """
    <style>
      /* ラベル行：固定高さ + flexで縦中央 + はみ出しクリップ */
      .labelrow{
        height: 32px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        color: rgba(49, 51, 63, 0.95);
        margin-bottom: 2px;
        overflow: hidden;
      }

      .top3tag {
        flex: 0 0 auto;
        font-size: 0.78rem;
        font-weight: 800;
        color: rgba(120, 70, 0, 1);
        background: rgba(255, 165, 0, 0.25);
        padding: 2px 10px;
        border-radius: 999px;
        line-height: 1;
      }

      /* 説明文：常に出す（空でも高さ確保） */
      .helptext {
        min-height: 18px;
        color: rgba(49, 51, 63, 0.55);
        font-size: 0.84rem;
        margin-top: -6px;
        margin-bottom: 6px;
      }

      /* 目安：2行想定で固定高（row内の見た目も安定） */
      .hintbox {
        height: 44px;
        line-height: 1.25;
        color: rgba(49, 51, 63, 0.6);
        font-size: 0.85rem;
        margin-top: -2px;
        margin-bottom: 12px;
        overflow: hidden;
      }

      .reason {
        background: rgba(240, 242, 246, 0.85);
        border-radius: 12px;
        padding: 12px 14px;
      }
      .reason ul { margin: 0.3rem 0 0.2rem 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

def label_line(label: str, is_top3: bool):
    if is_top3:
        st.markdown(
            f'<div class="labelrow"><span>{label}</span><span class="top3tag">TOP3</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="labelrow"><span>{label}</span></div>',
            unsafe_allow_html=True
        )

def helptext(text: str | None):
    # 空でも高さを確保する（ズレ抑止）
    t = text if text else "&nbsp;"
    st.markdown(f'<div class="helptext">{t}</div>', unsafe_allow_html=True)

def hint(text: str):
    st.markdown(f'<div class="hintbox">{text}</div>', unsafe_allow_html=True)

def number_input_block(
    key: str,
    label: str,
    is_top3: bool,
    input_kwargs: dict,
    hint_text: str,
    help_text: str | None = None,
):
    # 1) ラベル（HTML）
    label_line(label, is_top3)

    # 2) 入力（labelは隠す）
    val = st.number_input(
        label="__hidden__" + key,
        key=key,
        label_visibility="collapsed",
        **input_kwargs,
    )

    # 3) 説明 + 目安（高さ確保）
    helptext(help_text)
    hint(hint_text)

    return val

# ==================================================
# パス（起動前にプロジェクト直下へ cd 推奨）
# ==================================================
MODEL_FILE = "diabetes_xgb_model.pkl"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"

BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)
FI_PATH = os.path.join(BASE_DIR, FEATURE_IMPORTANCE_FILE)

# ==================================================
# 特徴量名（日本語）
# ==================================================
FEATURE_NAME_JA = {
    "Pregnancies": "妊娠回数",
    "Glucose": "血糖値",
    "BloodPressure": "血圧（拡張期）",
    "SkinThickness": "皮下脂肪の厚み",
    "Insulin": "インスリン",
    "BMI": "BMI（体格指数）",
    "DiabetesPedigreeFunction": "家族歴スコア（DPF）",
    "Age": "年齢",
}

# 理由用の簡易目安（学習用）
REASON_HINTS = {
    "Glucose": {"hi": 140, "mid": 110},
    "BMI": {"hi": 30, "mid": 25},
    "Age": {"hi": 45, "mid": 35},
    "BloodPressure": {"hi": 90, "mid": 80},
    "SkinThickness": {"hi": 50, "mid": 30},
    "Insulin": {"hi": 200, "mid": 120},
    "DiabetesPedigreeFunction": {"hi": 0.8, "mid": 0.5},
    "Pregnancies": {"hi": 6, "mid": 3},
}

# ==================================================
# しきい値（サイドバー）
# ==================================================
st.sidebar.header("設定")
threshold = st.sidebar.slider(
    "判定しきい値（この値以上なら「高め」）",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)
# ★ここを追加：％表示
st.sidebar.markdown(f"**現在のしきい値：{threshold*100:.0f}%**")
st.sidebar.caption("※しきい値を上げるほど「高め」と判定されにくくなります。")

# ==================================================
# 読み込み
# ==================================================
@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"学習済みモデルが見つかりません: {MODEL_PATH}\n"
            "（ヒント）Colabなら、プロジェクト直下に %cd してから起動してください。"
        )
        st.stop()

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact.get("model")
        feature_names = artifact.get("feature_names")
        imputer = artifact.get("imputer", None)
    else:
        model = artifact
        feature_names = None
        imputer = None

    if feature_names is None:
        feature_names = list(FEATURE_NAME_JA.keys())

    return model, feature_names, imputer

@st.cache_data
def load_feature_importance():
    if not os.path.exists(FI_PATH):
        return None
    df = pd.read_csv(FI_PATH)
    cols = [c.lower() for c in df.columns]
    if "feature" in cols and "importance" in cols:
        fcol = df.columns[cols.index("feature")]
        icol = df.columns[cols.index("importance")]
        out = df[[fcol, icol]].rename(columns={fcol: "feature", icol: "importance"})
    else:
        # 最低限2列想定
        if df.shape[1] < 2:
            return None
        out = df.iloc[:, :2].copy()
        out.columns = ["feature", "importance"]

    out["importance"] = pd.to_numeric(out["importance"], errors="coerce")
    out = out.dropna(subset=["importance"]).sort_values("importance", ascending=False)
    return out

def get_top3_features(fi_df: pd.DataFrame | None) -> list[str]:
    if fi_df is None or fi_df.empty or "feature" not in fi_df.columns:
        return ["Glucose", "BMI", "Age"]
    return fi_df.head(3)["feature"].astype(str).tolist()

def build_reason_lines(top3: list[str], raw_values: dict) -> list[str]:
    lines = []
    for f in top3:
        ja = FEATURE_NAME_JA.get(f, f)
        v = raw_values.get(f, None)

        if v is None or (isinstance(v, float) and np.isnan(v)):
            lines.append(f"{ja} は未測定（0入力）として扱われています。推定の不確実性が上がります。")
            continue

        cfg = REASON_HINTS.get(f)
        if cfg is None:
            lines.append(f"{ja} は重要特徴量に含まれるため、予測へ影響しています。")
            continue

        hi, mid = cfg["hi"], cfg["mid"]
        val = float(v)

        if val >= hi:
            lines.append(f"{ja} が高め（入力値: {val:g}）のため、確率を押し上げる方向に働きやすいです。")
        elif val >= mid:
            lines.append(f"{ja} がやや高め（入力値: {val:g}）のため、確率に影響している可能性があります。")
        else:
            lines.append(f"{ja} は極端に高くはない（入力値: {val:g}）ため、強い押し上げ要因ではなさそうです。")

    return lines[:3]

model, feature_names, imputer = load_artifact()
fi_df = load_feature_importance()
top3_features = get_top3_features(fi_df)

# ==================================================
# 用語の説明
# ==================================================
with st.expander("用語の説明（クリックで開く）"):
    st.write("**皮下脂肪の厚み（SkinThickness）**：上腕の皮下脂肪の厚み（mm）の指標。0は未測定扱い。")
    st.write("**家族歴スコア（DPF）**：糖尿病の家族歴・遺伝要因を表すスコア（単位なし）。")
    st.write("**血糖値/血圧/インスリン**：データセット上の指標。0は未測定扱い。")

# ==================================================
# 入力UI（★ここが重要：row単位で描画 → 絶対にズレない）
# ==================================================
st.subheader("① 入力項目")
st.caption("※学習用データセットの指標に基づく入力です")

# 各入力の仕様（ラベル・説明・目安）
spec = {
    "Pregnancies": dict(
        label="妊娠回数（Pregnancies）",
        input_kwargs=dict(min_value=0, max_value=20, value=0),
        help_text="男性の場合は0でOKです。",
        hint_text="目安：0〜10程度（個人差あり）",
    ),
    "Glucose": dict(
        label="血糖値（Glucose）",
        input_kwargs=dict(min_value=0, max_value=300, value=120),
        help_text="血糖値の指標です。0は未測定扱いとして扱います。",
        hint_text="目安：70〜140あたり（データセット由来）。0は未測定扱い。",
    ),
    "BloodPressure": dict(
        label="拡張期血圧（BloodPressure）",
        input_kwargs=dict(min_value=0, max_value=200, value=70),
        help_text="拡張期血圧（下の血圧）を想定。0は未測定扱い。",
        hint_text="目安：60〜90あたり。0は未測定扱い。",
    ),
    "SkinThickness": dict(
        label="皮下脂肪の厚み（SkinThickness）",
        input_kwargs=dict(min_value=0, max_value=100, value=20),
        help_text="上腕の皮下脂肪の厚み（mm）の指標。0は未測定扱い。",
        hint_text="目安：10〜50mm程度。0は未測定扱い。",
    ),
    "Insulin": dict(
        label="インスリン（Insulin）",
        input_kwargs=dict(min_value=0, max_value=900, value=80),
        help_text="インスリン値の指標。0は未測定扱い。",
        hint_text="目安：データのばらつき大。0は未測定扱い。",
    ),
    "BMI": dict(
        label="BMI（体格指数）",
        input_kwargs=dict(min_value=0.0, max_value=70.0, value=26.0, step=0.1),
        help_text="体格指数です。",
        hint_text="目安：18.5〜25が標準、25以上は高め。",
    ),
    "DiabetesPedigreeFunction": dict(
        label="家族歴スコア（DPF）",
        input_kwargs=dict(min_value=0.0, max_value=3.0, value=0.5, step=0.05),
        help_text="DiabetesPedigreeFunction：糖尿病の家族歴・遺伝要因を表すスコア（データセット内指標）",
        hint_text="目安：0.0〜2.5程度（スコアなので単位はありません）",
    ),
    "Age": dict(
        label="年齢（Age）",
        input_kwargs=dict(min_value=10, max_value=100, value=40),
        help_text=None,
        hint_text="目安：10〜100",
    ),
}

# row（左,右）の並び：ここを変えるとUIの並びも変わる
rows = [
    ("Pregnancies", "Insulin"),
    ("Glucose", "BMI"),
    ("BloodPressure", "DiabetesPedigreeFunction"),
    ("SkinThickness", "Age"),
]

# 値を格納
inputs = {}

for left_key, right_key in rows:
    c1, c2 = st.columns(2)

    with c1:
        s = spec[left_key]
        inputs[left_key] = number_input_block(
            key=left_key,
            label=s["label"],
            is_top3=(left_key in top3_features),
            input_kwargs=s["input_kwargs"],
            hint_text=s["hint_text"],
            help_text=s["help_text"],
        )

    with c2:
        s = spec[right_key]
        inputs[right_key] = number_input_block(
            key=right_key,
            label=s["label"],
            is_top3=(right_key in top3_features),
            input_kwargs=s["input_kwargs"],
            hint_text=s["hint_text"],
            help_text=s["help_text"],
        )

# ==================================================
# 前処理（0は欠損扱い）
# ==================================================
values = {
    "Pregnancies": inputs["Pregnancies"],
    "Glucose": np.nan if inputs["Glucose"] == 0 else inputs["Glucose"],
    "BloodPressure": np.nan if inputs["BloodPressure"] == 0 else inputs["BloodPressure"],
    "SkinThickness": np.nan if inputs["SkinThickness"] == 0 else inputs["SkinThickness"],
    "Insulin": np.nan if inputs["Insulin"] == 0 else inputs["Insulin"],
    "BMI": np.nan if inputs["BMI"] == 0 else inputs["BMI"],
    "DiabetesPedigreeFunction": inputs["DiabetesPedigreeFunction"],
    "Age": inputs["Age"],
}

if (
    inputs["Glucose"] == 0
    or inputs["BloodPressure"] == 0
    or inputs["SkinThickness"] == 0
    or inputs["Insulin"] == 0
    or inputs["BMI"] == 0
):
    st.info("補足：0は「未測定（欠損）」として扱われる前提の項目があります。0を入れると欠損扱いになる想定です。")

with st.expander("入力内容の確認（クリックで開く）"):
    confirm_df = pd.DataFrame(
        {"項目（英語）": list(values.keys()), "入力値": [values[k] for k in values.keys()]}
    )
    confirm_df["項目（日本語）"] = confirm_df["項目（英語）"].map(FEATURE_NAME_JA).fillna(confirm_df["項目（英語）"])
    confirm_df = confirm_df[["項目（日本語）", "項目（英語）", "入力値"]]
    st.dataframe(confirm_df, use_container_width=True)

# ==================================================
# 予測
# ==================================================
if st.button("② この条件で判定する"):
    X = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)

    if imputer is not None:
#       X = pd.DataFrame(imputer.transform(X), columns=feature_names)
        X = pd.DataFrame(imputer.transform(X), columns=imputer.feature_names_in_)

    else:
        X = X.fillna(0)

    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= threshold)

    st.subheader("② 判定結果")
    st.metric("糖尿病である可能性（予測確率）", f"{proba * 100:.1f} %")
    st.caption(f"判定しきい値：{threshold:.2f}（この値以上なら「高め」）")

    if pred == 1:
        st.warning("モデル予測では「糖尿病の可能性が高め」側に分類されました。")
    else:
        st.success("モデル予測では「可能性が低め」側に分類されました。")

    # TOP3（日本語表記）
    st.subheader("（参考）モデルが重視した特徴量 TOP3")
    if fi_df is None or fi_df.empty:
        st.info("feature_importance.csv が見つからない／形式が読めなかったため、代表TOP3（Glucose/BMI/Age）で表示します。")
        top3 = ["Glucose", "BMI", "Age"]
        top3_table = pd.DataFrame({"特徴量": [FEATURE_NAME_JA.get(x, x) for x in top3], "重要度": ["-", "-", "-"]})
        st.table(top3_table)
    else:
        top3_df = fi_df.head(3).reset_index(drop=True)
        top3_df["特徴量"] = top3_df["feature"].map(FEATURE_NAME_JA).fillna(top3_df["feature"])
        top3_df["重要度"] = pd.to_numeric(top3_df["importance"], errors="coerce").round(4)
        st.table(top3_df[["特徴量", "重要度"]])
        top3 = top3_df["feature"].astype(str).tolist()

    # 理由（自然文）
    
    st.subheader("（参考）今回の入力で、なぜこの結果になりそうか（学習用）")
    reason_lines = build_reason_lines(top3, values)
    st.markdown(
        '<div class="reason">TOP3（重要特徴量）に基づき、入力値から推測される要因をまとめました。'
        "<ul>"
        + "".join([f"<li>{line}</li>" for line in reason_lines])
        + "</ul></div>",
        unsafe_allow_html=True,
    )

    # 生活習慣ヒント
    st.subheader("③ 生活習慣のヒント（学習用）")
    advice = []

    g = inputs["Glucose"]
    b = inputs["BMI"]
    a = inputs["Age"]
    d = inputs["DiabetesPedigreeFunction"]

    if g >= 140:
        advice.append("・血糖値が高めです。甘い飲み物や間食を控えることを意識してみましょう。")
    elif g >= 110:
        advice.append("・血糖値がやや高めです。食べる順番（野菜→たんぱく質→炭水化物）を意識すると安心です。")

    if b >= 30:
        advice.append("・BMIが高めです。無理のない範囲での運動習慣（散歩など）がおすすめです。")
    elif b >= 25:
        advice.append("・BMIがやや高めです。夜遅い食事や砂糖入り飲料を控えてみましょう。")

    if a >= 45:
        advice.append("・年齢とともにリスクは上がりやすいため、定期的な健康診断が安心です。")

    if d >= 0.8:
        advice.append("・ご家族に糖尿病の方がいる場合、早めの生活習慣見直しが有効なことがあります。")

    if not advice:
        advice.append("・大きなリスクは見えにくいですが、バランスの良い食事と適度な運動を継続しましょう。")

    for line in advice:
        st.write(line)

    st.caption("※本アプリは学習用のデモです。実際の診断や治療は医療機関にご相談ください。")

# ==================================================
# 展望（今回未実装：発表用メモ）
# ==================================================
# 1) 入力妥当性チェック（極端値の警告、未測定のガイド強化）
# 2) 確率の直感表示（ゲージ/低中高の帯表示）
# 3) 説明性（SHAPなどで“今回の入力”の寄与を可視化）
# 4) 運用性（サンプル入力ボタン、入力ログDL、再現性向上）
# 5) データ限界の明示（母集団・偏り・目的外利用の注意）
