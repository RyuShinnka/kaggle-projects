import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")




# | 类型             | 用什么填           |
# | -------------- | -------------- |
# | 数值型（Age, Fare） | 中位数 `median()` |
# | 类别型（Embarked）  | 众数 `mode()[0]` |

train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

sex_map = {"male":0, "female":1}
train["Sex"] = train["Sex"].map(sex_map)
test["Sex"] = test["Sex"].map(sex_map)

embarked_map = {"S":0, "C":1, "Q":2}
train["Embarked"] = train["Embarked"].map(embarked_map)
test["Embarked"] = test["Embarked"].map(embarked_map)

# モデルに使う特徴量（Pclass, Sex, Age, Fare）を訓練用データから取り出す
X_train = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]

# 正解ラベル（Survived）も取り出す
y_train = train["Survived"]
# テストデータからも同じ特徴量を取り出す
X_test = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]]

# RandomForestClassifierモデルを作成する
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
# 学習用データ（X_train, y_train）を使ってモデルを訓練する
model.fit(X_train, y_train)
# テストデータ（X_test）を使って予測を行う
y_pred = model.predict(X_test)

# PassengerId と予測結果 y_pred を組み合わせて DataFrame を作成する
# カラム名は "PassengerId", "Survived" にする
df = pd.DataFrame({
    "PassengerId": test["PassengerId"]
    ,"Survived": y_pred
})


# ファイル名 "submission.csv" として保存する
df.to_csv("submissions/randomforest_add_features_submission.csv", index = False)


