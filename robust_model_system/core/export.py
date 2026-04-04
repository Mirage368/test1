# 结果导出核心模块
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def export_data(df, save_path, file_name="processed_data"):
    """导出预处理后的数据为CSV"""
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_path = os.path.join(save_path, f"{file_name}.csv")
        df.to_csv(full_path, index=False, encoding="utf-8")
        return {"status": "success", "msg": f"数据已导出至：{full_path}"}
    except Exception as e:
        return {"status": "error", "msg": f"数据导出失败：{str(e)}"}

def export_model(model, save_path, model_name="best_model"):
    """导出训练好的模型为pickle文件"""
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        full_path = os.path.join(save_path, f"{model_name}.pkl")
        with open(full_path, "wb") as f:
            pickle.dump(model, f)
        return {"status": "success", "msg": f"模型已导出至：{full_path}"}
    except Exception as e:
        return {"status": "error", "msg": f"模型导出失败：{str(e)}"}

def export_metrics(metrics_dict, save_path, file_name="model_metrics"):
    """导出模型性能指标为CSV"""
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 转换为DataFrame
        df_metrics = pd.DataFrame(metrics_dict).T  
        full_path = os.path.join(save_path, f"{file_name}.csv")
        df_metrics.to_csv(full_path, encoding="utf-8")
        return {"status": "success", "msg": f"指标已导出至：{full_path}"}
    except Exception as e:
        return {"status": "error", "msg": f"指标导出失败：{str(e)}"}
