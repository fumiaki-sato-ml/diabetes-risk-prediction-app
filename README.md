# Diabetes Risk Prediction App（糖尿病リスク予測アプリ）

Streamlitで動作する **糖尿病リスク推定（2値分類）Webアプリ**です。  
学習用データセットの指標（Pregnancies / Glucose / BMI など）を入力すると、推定結果を表示します。

- 公開URL： https://diabetes-risk-prediction-app-ac3wsvidihz9gyjcd6wswt.streamlit.app/
- GitHub： https://github.com/fumiaki-sato-ml/diabetes-risk-prediction-app/tree/master

---

## 目的（なぜ作ったか）
- MLモデルを「作る」だけでなく、**入力 → 推定 → 結果表示**までを一通り体験できる形で実装するため
- 実務のDX文脈を意識し、**第三者が触って理解できるUI**（入力説明・目安・注意書き）を用意するため

---

## できること
- 入力フォームから指標を入力して、糖尿病リスクを推定
- 入力値の目安（レンジ）や注意点を画面上で案内（例：0は未測定扱い など）

---

## 入力項目

> ※学習用データセットの指標に基づく入力です  
> ※一部項目は「0を未測定扱い」として扱います（画面の注意書きに準拠）

|項目|英語名|説明|目安・補足|
|---|---|---|---|
|妊娠回数|Pregnancies|妊娠回数|男性の場合は0でOK。目安：0〜10程度（個人差あり）|
|インスリン|Insulin|インスリン値の指標|**0は未測定扱い**。目安：データのばらつき大（0は未測定）|
|血糖値|Glucose|血糖値の指標|**0は未測定扱い**。目安：70〜140あたり（データセット由来）|
|BMI|BMI|体格指数|目安：18.5〜25が標準、25以上は高め|
|拡張期血圧|BloodPressure|拡張期血圧（下の血圧）|**0は未測定扱い**。目安：60〜90あたり（0は未測定）|
|家族歴スコア|DiabetesPedigreeFunction（DPF）|糖尿病の家族歴・遺伝要因を表すスコア|目安：0.0〜2.5程度（スコアのため単位なし）|
|皮下脂肪の厚み|SkinThickness|上腕の皮下脂肪の厚み（mm）|**0は未測定扱い**。目安：10〜50mm程度（0は未測定）|
|年齢|Age|年齢|目安：10〜100|

---

## 使い方（公開URL）
1. 公開URLにアクセス  
2. 各入力項目を入力（未測定の場合は0を入力する項目あり）  
3. 予測を実行  
4. 結果が画面に表示されます

---

## ローカル実行方法

### 1) セットアップ
```bash
git clone https://github.com/fumiaki-sato-ml/diabetes-risk-prediction-app.git
cd diabetes-risk-prediction-app
pip install -r requirements.txt
