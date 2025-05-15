import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# train.csv と test.csv を読み込む（data/ フォルダから）
train_path = "data/train.csv"
test_path = "data/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
# train の形状・先頭の数行を確認する
# print(train.head())
# SalePrice（目的変数）のヒストグラムを描いて、分布を確認する
# log変換前の分布
# train["SalePrice"].hist(bins=50)
# plt.title("Before log")
# plt.show()
# SalePrice に log1p 変換を適用する（歪みを小さくするため）
train["SalePrice"] = np.log1p(train["SalePrice"])

# log変換後の分布
# train["SalePrice"].hist(bins=50)
# plt.title("After log")
# plt.show()

# 欠損値（NaN）の合計を列ごとに確認する（train）
# print(train.isnull().sum())
# 欠損の多い順に表示（.isnull().sum().sort_values(ascending=False)）
# print(train.isnull().sum().sort_values(ascending=False))

# 欠損割合（欠損数 / データ数）を計算して確認する
# result = train.isnull().sum() / train.shape[0]
# print(result.sort_values(ascending=False))

# PoolQC, MiscFeature, Alley, Fence など欠損率が高すぎる列は削除する
train.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType"], axis= 1)
test.drop(["PoolQC", "MiscFeature", "Alley", "Fence",  "MasVnrType"], axis= 1)
# それ以外の欠損列は、型に応じて median または "None"/mode() で補完する
fillMedian= ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
fillNone = ["GarageType", "BsmtFinType2", "BsmtFinType1", "BsmtExposure", "BsmtCond", "BsmtQual", "GarageCond", "GarageQual", "GarageFinish"]

for z in fillMedian:
    train[z] = train[z].fillna(train[z].median())
    test[z] = test[z].fillna(test[z].median())

for y in fillNone:
    train[y] = train[y].fillna("None")
    test[y] = test[y].fillna("None")
train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])
test["Electrical"] = test["Electrical"].fillna(test["Electrical"].mode()[0])

# 文字列（object 型）の特徴量を LabelEncoder で数値に変換する

# 1. object 型の列名をすべて抽出する
object_cols = train.select_dtypes(include= "object").columns


# 2. for 文で各列を順番に処理する
for col in object_cols:
    le =  LabelEncoder()
    # 3. train[col] と test[col] を結合し、LabelEncoder で fit する
    le.fit(pd.concat([train[col], test[col]]))
    # 4. その encoder を使って train[col], test[col] を transform する
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# 新しい特徴量を追加する
# TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF として作成する
train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]
test["TotalSF"] = test["1stFlrSF"] + test["2ndFlrSF"] + test["TotalBsmtSF"]

# 目的変数 y（SalePrice）を取り出す
y_train = train["SalePrice"]
# 学習用の特徴量 X を作成する（Id 列や SalePrice を除く）
X_train = train.drop(["Id", "SalePrice"], axis=1)
# print(X_train.isnull().sum())

# ランダムフォレストモデルを作成する（max_depth=5, random_state=0）
model = RandomForestRegressor(max_depth=5, random_state=0)
# 学習データ（X_train, y_train）を使ってモデルを訓練する
model.fit(X_train, y_train)
# test データも同じ列（X_test）を作って予測を行う
X_test = test[X_train.columns]
y_pred = model.predict(X_test)
# 予測値を expm1 で log を元に戻す
y_pred_real = np.expm1(y_pred)
# sample_submission.csv に予測結果を書き込み、提出ファイルを保存する
df = pd.DataFrame({
    "Id": test["Id"]
    ,"SalePrice":y_pred_real
})
df.to_csv("submissions/submission_add_features.csv", index= False)
print("submission.csv 保存完了")