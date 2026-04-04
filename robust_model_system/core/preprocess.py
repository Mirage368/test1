import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from pandas.api.types import is_numeric_dtype


def load_data(uploaded_file):
    """加载上传的CSV文件，返回DataFrame"""
    try:
        df = pd.read_csv(uploaded_file)
        return {"status": "success", "data": df, "msg": "数据加载成功"}
    except Exception as e:
        return {"status": "error", "data": None, "msg": f"数据加载失败：{str(e)}"}


def process_missing(df):
    """处理缺失值：数值型用中位数，分类型用众数"""
    df_processed = df.copy()

    for col in df_processed.columns:
        if is_numeric_dtype(df_processed[col]):
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
        else:
            mode_vals = df_processed[col].mode(dropna=True)
            if len(mode_vals) > 0:
                df_processed[col] = df_processed[col].fillna(mode_vals[0])
            else:
                df_processed[col] = df_processed[col].fillna("Unknown")

    return df_processed


def process_outlier(df, exclude_cols=None):
    """处理异常值：IQR方法截断，仅处理数值型列，可排除指定列"""
    df_processed = df.copy()
    exclude_cols = exclude_cols or []

    numeric_cols = df_processed.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col in exclude_cols:
            continue

        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0 or pd.isna(IQR):
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)

    return df_processed


def standardize_data(df, exclude_cols=None):
    """数据标准化：RobustScaler，仅处理数值型列，可排除指定列"""
    df_scaled = df.copy()
    exclude_cols = exclude_cols or []

    numeric_cols = [
        col for col in df_scaled.select_dtypes(include=np.number).columns
        if col not in exclude_cols
    ]

    scaler = None
    if len(numeric_cols) > 0:
        scaler = RobustScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    return df_scaled, scaler


def run_preprocess(
    df,
    do_missing=True,
    do_outlier=True,
    do_scale=True,
    exclude_cols=None
):
    """
    按用户选择执行预处理流程
    exclude_cols: 不参与异常值处理/标准化的列，例如回归目标列
    """
    df_processed = df.copy()
    exclude_cols = exclude_cols or []
    scaler = None

    if do_missing:
        df_processed = process_missing(df_processed)

    if do_outlier:
        df_processed = process_outlier(df_processed, exclude_cols=exclude_cols)

    if do_scale:
        df_processed, scaler = standardize_data(df_processed, exclude_cols=exclude_cols)

    return df_processed, scaler
