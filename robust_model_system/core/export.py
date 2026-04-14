import os
import io
import pickle
import pandas as pd


def export_dataframe_to_csv_bytes(df):
    """将DataFrame转换为可下载的CSV二进制数据"""
    return df.to_csv(index=False).encode("utf-8-sig")


def export_summary_to_csv_bytes(summary_dict):
    """将摘要字典转换为CSV二进制数据"""
    df_summary = pd.DataFrame([summary_dict])
    return df_summary.to_csv(index=False).encode("utf-8-sig")


def export_model_to_bytes(model):
    """将模型对象序列化为二进制"""
    model_buffer = io.BytesIO()
    pickle.dump(model, model_buffer)
    model_buffer.seek(0)
    return model_buffer


def save_dataframe_to_local(df, save_path, file_name="processed_data.csv"):
    """保存DataFrame到本地"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, file_name)
    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    return full_path


def save_model_to_local(model, save_path, file_name="best_model.pkl"):
    """保存模型到本地"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    full_path = os.path.join(save_path, file_name)
    with open(full_path, "wb") as f:
        pickle.dump(model, f)
    return full_path
