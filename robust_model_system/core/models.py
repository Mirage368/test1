import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score

# ==============================
# 模型定义
# ==============================
SUPPORTED_MODELS = {
    "回归": {
        "线性回归": LinearRegression(),
        "鲁棒回归(RANSAC)": RANSACRegressor(random_state=42),
        "随机森林回归": RandomForestRegressor(random_state=42),
        "梯度提升回归": GradientBoostingRegressor(random_state=42)
    },
    "聚类": {
        "KMeans": KMeans(n_clusters=3, random_state=42, n_init=10),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
    }
}


# ==============================
# 单模型训练：回归
# ==============================
def train_single_model(X, y, model_name):
    model = SUPPORTED_MODELS["回归"][model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "R2": round(r2_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "CV_R2": round(cross_val_score(model, X, y, cv=5, scoring="r2").mean(), 4)
    }

    return metrics, model


# ==============================
# 单模型训练：聚类
# ==============================
def train_single_cluster_model(X, model_name):
    model = SUPPORTED_MODELS["聚类"][model_name]

    labels = model.fit_predict(X)
    unique_labels = set(labels)

    if len(unique_labels) <= 1:
        silhouette = None
    else:
        effective_labels = set(labels)
        if -1 in effective_labels:
            effective_labels.remove(-1)

        if len(effective_labels) < 2:
            silhouette = None
        else:
            try:
                silhouette = round(silhouette_score(X, labels), 4)
            except Exception:
                silhouette = None

    metrics = {
        "Model": model_name,
        "Silhouette": silhouette
    }

    return metrics, model


# ==============================
# 总训练与评估函数
# ==============================
def train_and_evaluate(X, y, model_type):
    results = []
    trained_models = {}

    # 回归任务
    if model_type == "回归":
        for model_name in SUPPORTED_MODELS["回归"].keys():
            try:
                res, model = train_single_model(X, y, model_name)
                results.append(res)
                trained_models[model_name] = model
            except Exception:
                results.append({
                    "Model": model_name,
                    "R2": None,
                    "MAE": None,
                    "RMSE": None,
                    "CV_R2": None
                })

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="CV_R2", ascending=False, na_position="last").reset_index(drop=True)
        return df_results, trained_models

    # 聚类任务
    elif model_type == "聚类":
        for model_name in SUPPORTED_MODELS["聚类"].keys():
            try:
                res, model = train_single_cluster_model(X, model_name)
                results.append(res)
                trained_models[model_name] = model
            except Exception:
                results.append({
                    "Model": model_name,
                    "Silhouette": None
                })

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="Silhouette", ascending=False, na_position="last").reset_index(drop=True)
        return df_results, trained_models

    # 其他类型
    else:
        df_results = pd.DataFrame([{"Model": "不支持该任务类型"}])
        return df_results, trained_models
