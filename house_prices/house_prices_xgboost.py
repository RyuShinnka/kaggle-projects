# ------------------------------
# Step 0: 必要なライブラリをインポートする
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
# ------------------------------
# Step 1: データを読み込む
train_path = "data/train.csv"
test_path = "data/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
# ------------------------------
# Step 2: 目的変数（SalePrice）を log1p 変換する
train["SalePrice"] = np.log1p(train["SalePrice"])
# ------------------------------
# Step 3: 欠損値を処理する
train.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType"], axis=1)
test.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType"], axis=1)

fill_median = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]
fill_none = ["GarageType", "BsmtFinType2", "BsmtFinType1", "BsmtExposure", "BsmtCond", "BsmtQual", "GarageCond", "GarageQual", "GarageFinish"]

for feature in fill_median:
    train[feature] = train[feature].fillna(train[feature].median())
    test[feature] = test[feature].fillna(test[feature].median())

for feature in fill_none:
    train[feature] = train[feature].fillna("None")
    test[feature] = test[feature].fillna("None")
train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])
test["Electrical"] = test["Electrical"].fillna(test["Electrical"].mode()[0])
# ------------------------------
# Step 4: 特徴量を追加する
train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]
test["TotalSF"] = test["1stFlrSF"] + test["2ndFlrSF"] + test["TotalBsmtSF"]

train["TotalBath"] = train["FullBath"] + 0.5 * train["HalfBath"] + train["BsmtFullBath"] + 0.5 * train["BsmtHalfBath"]
test["TotalBath"] = test["FullBath"] + 0.5 * test["HalfBath"] + test["BsmtFullBath"] + 0.5 * test["BsmtHalfBath"]

# ------------------------------
# Step 5: ラベルエンコーディングを行う
object_cols = train.select_dtypes(include="object").columns
for col in object_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
# ------------------------------
# Step 6: 説明変数（X）と目的変数（y）を作成する
X_train = train.drop(["Id", "SalePrice"], axis=1)
X_test = test[X_train.columns]
y_train = train["SalePrice"]
# ------------------------------
# Step 7: XGBoost モデルを学習する
model = XGBRegressor(
    max_depth=4,
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=0
)
model.fit(X_train, y_train)
# ------------------------------
# Step 8: テストデータで予測し、log を元に戻す
y_pred = model.predict(X_test)
y_pred_real = np.expm1(y_pred)
# ------------------------------
# Step 9: 提出ファイルを作成する
# - Id と SalePrice を列に持つ DataFrame を作成
# - submissions/submission_xgboost.csv として保存する
df = pd.DataFrame({
    "Id": test["Id"]
    ,"SalePrice":y_pred_real
})
df.to_csv("submissions/submission_xgboost_tuned.csv", index=False)
print("csvファイル 保存完了")
