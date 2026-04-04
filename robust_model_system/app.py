import streamlit as st
import pandas as pd
import time
from core.preprocess import load_data, run_preprocess
from core.models import train_and_evaluate

st.set_page_config(page_title="鲁棒模型拟合系统", layout="wide")

st.title("鲁棒模型拟合系统")

# ==============================
# 上传数据
# ==============================
uploaded_file = st.file_uploader("上传CSV数据文件", type=["csv"])

if uploaded_file:
    data_result = load_data(uploaded_file)

    if data_result["status"] == "success":
        df_original = data_result["data"]

        st.success("数据加载成功！")

        # ==============================
        # 原始数据展示
        # ==============================
        st.subheader("原始数据预览")
        st.dataframe(df_original.head(10))

        st.write("数据基本信息：")
        st.write(df_original.describe())

        # ==============================
        # 模型类型选择
        # ==============================
        st.subheader("模型设置")

        model_type = st.selectbox(
            "选择任务类型",
            ["无", "回归", "聚类"]
        )

        target_col = None

        if model_type == "回归":
            target_col = st.selectbox(
                "选择目标变量（标签列）",
                df_original.columns
            )
        elif model_type == "聚类":
            st.info("聚类任务无需选择目标变量，系统将直接对特征数据进行聚类分析。")

        # ==============================
        # 预处理配置
        # ==============================
        st.subheader("数据预处理配置")

        do_preprocess = st.checkbox("启用数据预处理", value=True)

        do_missing = False
        do_outlier = False
        do_scale = False

        if do_preprocess:
            do_missing = st.checkbox("处理缺失值（中位数/众数）", value=True)
            do_outlier = st.checkbox("处理异常值（IQR截断）", value=True)
            do_scale = st.checkbox("鲁棒标准化（RobustScaler）", value=True)

        # ==============================
        # 执行预处理
        # ==============================
        st.subheader("执行数据处理")

        if st.button("开始处理数据"):

            exclude_cols = []

            # 回归任务：目标列不参与异常值处理和标准化
            if model_type == "回归" and target_col:
                if target_col in df_original.columns:
                    exclude_cols = [target_col]

            with st.status("处理中...", expanded=True) as status:

                if do_preprocess:
                    st.write(f"缺失值处理：{'是' if do_missing else '否'}")
                    st.write(f"异常值处理：{'是' if do_outlier else '否'}")
                    st.write(f"标准化：{'是' if do_scale else '否'}")

                    if exclude_cols:
                        st.write(f"排除列（不参与异常值/标准化）：{exclude_cols}")
                else:
                    st.write("未启用预处理，直接使用原始数据")

                if do_preprocess:
                    df_processed, scaler = run_preprocess(
                        df=df_original,
                        do_missing=do_missing,
                        do_outlier=do_outlier,
                        do_scale=do_scale,
                        exclude_cols=exclude_cols
                    )
                else:
                    df_processed = df_original.copy()
                    scaler = None

                status.update(label="处理完成", state="complete")

            st.success("数据处理完成")

            st.subheader("处理后数据预览")
            st.dataframe(df_processed.head(10))

            # 保存到 session_state
            st.session_state["df_processed"] = df_processed
            st.session_state["scaler"] = scaler

        # ==============================
        # 数据可视化
        # ==============================
        if "df_processed" in st.session_state:
            df_processed = st.session_state["df_processed"]

            st.subheader("数据可视化")

            # 只提取数值型列，便于绘制散点图和直方图
            numeric_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns.tolist()

            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)

                with col1:
                    feature_x = st.selectbox(
                        "选择X轴变量",
                        numeric_cols,
                        key="vis_x"
                    )

                with col2:
                    default_y_index = 1 if len(numeric_cols) > 1 else 0
                    feature_y = st.selectbox(
                        "选择Y轴变量",
                        numeric_cols,
                        index=default_y_index,
                        key="vis_y"
                    )

                st.write("散点图：")

                # 仅保留所选两列，并去除缺失值
                scatter_df = df_processed[[feature_x, feature_y]].dropna().copy()

                if not scatter_df.empty:
                    # 显式指定 x 和 y，避免 Streamlit 默认把索引当横轴
                    scatter_df.columns = ["x", "y"]

                    st.scatter_chart(
                        scatter_df,
                        x="x",
                        y="y"
                    )
                else:
                    st.warning("所选的两列在去除缺失值后没有可用于绘图的数据。")

                st.write("分布图（直方图）：")
                feature_hist = st.selectbox(
                    "选择变量查看分布",
                    numeric_cols,
                    key="hist_feature"
                )

                hist_data = df_processed[feature_hist].dropna()

                if not hist_data.empty:
                    hist_counts = hist_data.value_counts().sort_index()
                    st.bar_chart(hist_counts)
                else:
                    st.warning("该变量没有可用于展示分布的数据。")

            elif len(numeric_cols) == 1:
                st.info("当前数据仅包含 1 个数值型字段，无法绘制双变量散点图。")

                st.write("分布图（直方图）：")
                feature_hist = numeric_cols[0]
                hist_data = df_processed[feature_hist].dropna()

                if not hist_data.empty:
                    hist_counts = hist_data.value_counts().sort_index()
                    st.bar_chart(hist_counts)
                else:
                    st.warning("该变量没有可用于展示分布的数据。")

            else:
                st.info("当前数据中没有可用于可视化的数值型列。")

        # ==============================
        # 模型训练
        # ==============================
        st.subheader("多模型训练与最优推荐")

        if "df_processed" in st.session_state and model_type != "无":

            df_processed = st.session_state["df_processed"]

            if st.button("开始训练模型"):

                start_time = time.time()

                with st.spinner("模型训练中..."):

                    # =========================
                    # 回归任务
                    # =========================
                    if model_type == "回归":
                        if target_col is not None and target_col in df_processed.columns:
                            X = df_processed.drop(columns=[target_col])
                            y = df_processed[target_col]

                            results = train_and_evaluate(X, y, model_type)

                            end_time = time.time()
                            runtime = end_time - start_time

                            st.success(f"训练完成！耗时：{runtime:.2f} 秒")
                            st.subheader("模型性能对比")
                            st.dataframe(results)

                            if not results.empty:
                                best_model = results.iloc[0]
                                st.success(f"推荐模型：{best_model['Model']}")
                        else:
                            st.error(f"目标列 {target_col} 不存在于处理后的数据中！")

                    # =========================
                    # 聚类任务
                    # =========================
                    elif model_type == "聚类":
                        X = df_processed.copy()

                        results = train_and_evaluate(X, None, model_type)

                        end_time = time.time()
                        runtime = end_time - start_time

                        st.success(f"训练完成！耗时：{runtime:.2f} 秒")
                        st.subheader("聚类模型性能对比")
                        st.dataframe(results)

                        if not results.empty:
                            best_model = results.iloc[0]
                            st.success(f"推荐模型：{best_model['Model']}")

        # ==============================
        # 导出数据
        # ==============================
        if "df_processed" in st.session_state:
            st.subheader("导出数据")

            df_to_export = st.session_state["df_processed"]
            csv = df_to_export.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="下载当前数据",
                file_name="processed_data.csv",
                data=csv,
                mime="text/csv"
            )

    else:
        st.error(data_result["msg"])
