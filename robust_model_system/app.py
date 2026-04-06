import streamlit as st
import pandas as pd
import time

from core.preprocess import load_data, run_preprocess
from core.models import train_and_evaluate


# ==============================
# 页面配置
# ==============================
st.set_page_config(
    page_title="鲁棒模型拟合系统",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================
# 少量全局样式：只调间距，不包文字内容
# ==============================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
div.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ==============================
# Session State 初始化
# ==============================
def init_session_state():
    defaults = {
        "logged_in": False,
        "df_original": None,
        "df_processed": None,
        "scaler": None,
        "uploaded_filename": None,
        "file_loaded": False,
        "model_results": None,
        "best_model_name": None,
        "runtime": None,
        "selected_model_type": "无",
        "selected_target_col": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==============================
# 公共方法
# ==============================
def reset_data_related_states():
    """上传新文件后，重置与数据处理、训练相关的状态"""
    st.session_state["df_processed"] = None
    st.session_state["scaler"] = None
    st.session_state["model_results"] = None
    st.session_state["best_model_name"] = None
    st.session_state["runtime"] = None
    st.session_state["selected_model_type"] = "无"
    st.session_state["selected_target_col"] = None


def render_system_title():
    st.title("鲁棒模型拟合系统")
    st.caption("集数据上传、预处理、可视化与多模型训练的数据分析系统")


# ==============================
# 登录页面
# ==============================
def render_login():
    st.title("鲁棒模型拟合系统")
    st.subheader("用户登录")

    st.write("请输入用户名和密码后进入系统。")

    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_mid:
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")

        col1, col2 = st.columns(2)
        with col1:
            login_btn = st.button("登录")
        with col2:
            clear_btn = st.button("清空")

        if clear_btn:
            st.rerun()

        if login_btn:
            if username == "user" and password == "123456":
                st.session_state["logged_in"] = True
                st.success("登录成功，正在进入系统...")
                st.rerun()
            else:
                st.error("用户名或密码错误，请重新输入")

        st.info("默认测试账号：user；默认密码：123456")


# ==============================
# 侧边栏
# ==============================
def render_sidebar():
    st.sidebar.title("功能菜单")

    page = st.sidebar.radio(
        "请选择模块",
        [
            "首页",
            "数据上传",
            "原始数据展示",
            "数据预处理",
            "数据可视化",
            "模型训练",
            "数据导出",
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("数据上传入口")

    uploaded_file = st.sidebar.file_uploader(
        "上传 CSV 数据文件",
        type=["csv"],
        key="csv_uploader"
    )

    # 只在新文件上传时重新读取和重置
    if uploaded_file is not None:
        current_name = uploaded_file.name
        previous_name = st.session_state["uploaded_filename"]

        if (not st.session_state["file_loaded"]) or (current_name != previous_name):
            data_result = load_data(uploaded_file)

            if data_result["status"] == "success":
                st.session_state["df_original"] = data_result["data"]
                st.session_state["uploaded_filename"] = current_name
                st.session_state["file_loaded"] = True

                reset_data_related_states()
                st.sidebar.success("数据加载成功！")
            else:
                st.sidebar.error(data_result["msg"])
        else:
            st.sidebar.info(f"当前文件：{st.session_state['uploaded_filename']}")

    st.sidebar.markdown("---")
    st.sidebar.write("当前系统模块：")
    st.sidebar.write("1.数据上传")
    st.sidebar.write("2.原始数据展示")
    st.sidebar.write("3.数据预处理")
    st.sidebar.write("4.数据可视化")
    st.sidebar.write("5.模型训练")
    st.sidebar.write("6.数据导出")

    if st.sidebar.button("退出登录"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    return page


# ==============================
# 首页
# ==============================
def render_home():
    render_system_title()

    st.subheader("系统概览")

    col1, col2, col3 = st.columns(3)

    with col1:
        data_status = "已上传" if st.session_state["df_original"] is not None else "未上传"
        st.metric("数据状态", data_status)

    with col2:
        preprocess_status = "已完成" if st.session_state["df_processed"] is not None else "未处理"
        st.metric("预处理状态", preprocess_status)

    with col3:
        train_status = "已完成" if st.session_state["model_results"] is not None else "未训练"
        st.metric("训练状态", train_status)

    st.markdown("---")

    st.subheader("使用说明")
    st.markdown("""
1. 在左侧边栏上传 CSV 数据文件。  
2. 在“原始数据展示”模块中查看原始数据内容。  
3. 在“数据预处理”模块中完成缺失值、异常值和标准化等操作（也可以跳过预处理部分）。  
4. 在“数据可视化”模块中查看散点图和数据分布（可手动选择x轴、y轴变量）。  
5. 在“模型训练”模块中执行训练并获取最优模型推荐。  
6. 在“数据导出”模块中下载处理后的数据。  
""")

    if st.session_state["uploaded_filename"]:
        st.success(f"当前已加载文件：{st.session_state['uploaded_filename']}")


# ==============================
# 数据上传页
# ==============================
def render_upload_page():
    render_system_title()
    st.subheader("数据上传")

    st.info("请在左侧边栏上传 CSV 文件。上传成功后，系统会自动读取数据。")

    if st.session_state["df_original"] is not None:
        df = st.session_state["df_original"]
        st.success(f"当前已上传文件：{st.session_state['uploaded_filename']}")
        st.write(f"数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
    else:
        st.warning("当前尚未上传数据文件。")


# ==============================
# 原始数据展示
# ==============================
def render_data_preview():
    render_system_title()
    st.subheader("原始数据展示")

    df_original = st.session_state["df_original"]

    if df_original is None:
        st.warning("请先在左侧上传 CSV 数据文件。")
        return

    st.success("数据加载成功！")

    st.write("### 原始数据预览")
    st.dataframe(df_original.head(10), use_container_width=True)

    st.write("### 数据基本信息")
    st.write(f"数据维度：{df_original.shape[0]} 行 × {df_original.shape[1]} 列")
    st.write(f"字段名称：{list(df_original.columns)}")

    st.write("### 描述统计")
    st.dataframe(df_original.describe(include="all"), use_container_width=True)


# ==============================
# 数据预处理
# ==============================
def render_preprocess_page():
    render_system_title()
    st.subheader("数据预处理")

    df_original = st.session_state["df_original"]

    if df_original is None:
        st.warning("请先在左侧上传CSV数据文件。")
        return

    st.write("### 任务设置")

    model_type = st.selectbox(
        "选择任务类型",
        ["无", "回归", "聚类"],
        index=["无", "回归", "聚类"].index(st.session_state["selected_model_type"])
        if st.session_state["selected_model_type"] in ["无", "回归", "聚类"] else 0
    )
    st.session_state["selected_model_type"] = model_type

    target_col = None

    if model_type == "回归":
        default_index = 0
        if st.session_state["selected_target_col"] in df_original.columns:
            default_index = list(df_original.columns).index(st.session_state["selected_target_col"])

        target_col = st.selectbox(
            "选择目标变量（标签列）",
            df_original.columns,
            index=default_index
        )
        st.session_state["selected_target_col"] = target_col

    elif model_type == "聚类":
        st.info("聚类任务无需选择目标变量，系统将直接使用特征数据进行聚类分析。")
        st.session_state["selected_target_col"] = None

    st.write("### 预处理选项")

    do_preprocess = st.checkbox("启用数据预处理", value=True)

    do_missing = False
    do_outlier = False
    do_scale = False

    if do_preprocess:
        do_missing = st.checkbox("处理缺失值（中位数/众数填充）", value=True)
        do_outlier = st.checkbox("处理异常值（IQR截断）", value=True)
        do_scale = st.checkbox("鲁棒标准化（RobustScaler）", value=True)

    st.write("### 执行预处理")

    if st.button("开始处理数据"):
        exclude_cols = []

        if model_type == "回归" and target_col and target_col in df_original.columns:
            exclude_cols = [target_col]

        with st.spinner("数据处理中..."):
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

        st.session_state["df_processed"] = df_processed
        st.session_state["scaler"] = scaler
        st.session_state["model_results"] = None
        st.session_state["best_model_name"] = None
        st.session_state["runtime"] = None

        st.success("数据预处理完成！")

    if st.session_state["df_processed"] is not None:
        st.write("### 处理后数据预览")
        st.dataframe(st.session_state["df_processed"].head(10), use_container_width=True)


# ==============================
# 数据可视化
# ==============================
def render_visualization_page():
    render_system_title()
    st.subheader("数据可视化")

    df_processed = st.session_state["df_processed"]

    if df_processed is None:
        st.warning("请先完成数据预处理后再查看可视化结果。")
        return

    numeric_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            feature_x = st.selectbox("选择X轴变量", numeric_cols, key="vis_x")

        with col2:
            default_y_index = 1 if len(numeric_cols) > 1 else 0
            feature_y = st.selectbox("选择Y轴变量", numeric_cols, index=default_y_index, key="vis_y")

        st.write("### 散点图")
        scatter_df = df_processed[[feature_x, feature_y]].dropna().copy()

        if not scatter_df.empty:
            scatter_df.columns = ["x", "y"]
            st.scatter_chart(scatter_df, x="x", y="y")
        else:
            st.warning("所选列在去除缺失值后没有可用于绘图的数据。")

        st.write("### 分布图")
        feature_hist = st.selectbox("选择变量查看分布", numeric_cols, key="hist_feature")
        hist_data = df_processed[feature_hist].dropna()

        if not hist_data.empty:
            hist_counts = hist_data.value_counts().sort_index()
            st.bar_chart(hist_counts)
        else:
            st.warning("该变量没有可用于展示分布的数据。")

    elif len(numeric_cols) == 1:
        st.info("当前数据仅包含 1 个数值型字段，无法绘制双变量散点图。")

        feature_hist = numeric_cols[0]
        hist_data = df_processed[feature_hist].dropna()

        st.write("### 分布图")
        if not hist_data.empty:
            hist_counts = hist_data.value_counts().sort_index()
            st.bar_chart(hist_counts)
        else:
            st.warning("该变量没有可用于展示分布的数据。")
    else:
        st.warning("当前数据中没有可用于可视化的数值型列。")


# ==============================
# 模型训练
# ==============================
def render_training_page():
    render_system_title()
    st.subheader("模型训练与最优推荐")

    df_original = st.session_state["df_original"]
    df_processed = st.session_state["df_processed"]
    model_type = st.session_state["selected_model_type"]
    target_col = st.session_state["selected_target_col"]

    if df_original is None:
        st.warning("请先上传数据。")
        return

    if df_processed is None:
        st.warning("请先完成数据预处理。")
        return

    if model_type == "无":
        st.warning("请先在“数据预处理”模块中选择任务类型。")
        return

    st.info(f"当前任务类型：{model_type}")
    if model_type == "回归" and target_col is not None:
        st.info(f"当前目标变量：{target_col}")

    if st.button("开始训练模型"):
        start_time = time.time()

        with st.spinner("模型训练中..."):
            if model_type == "回归":
                if target_col is None or target_col not in df_processed.columns:
                    st.error("当前目标列无效，请重新到“数据预处理”模块中选择目标列。")
                    return

                X = df_processed.drop(columns=[target_col])
                y = df_processed[target_col]
                results = train_and_evaluate(X, y, model_type)

            else:
                X = df_processed.copy()
                results = train_and_evaluate(X, None, model_type)

        end_time = time.time()

        st.session_state["model_results"] = results
        st.session_state["runtime"] = end_time - start_time

        if results is not None and not results.empty:
            st.session_state["best_model_name"] = results.iloc[0]["Model"]
        else:
            st.session_state["best_model_name"] = None

        st.success("模型训练完成！")

    if st.session_state["model_results"] is not None:
        if st.session_state["runtime"] is not None:
            st.success(f"训练耗时：{st.session_state['runtime']:.2f} 秒")

        if model_type == "回归":
            st.write("### 模型性能对比")
        else:
            st.write("### 聚类模型性能对比")

        st.dataframe(st.session_state["model_results"], use_container_width=True)

        if st.session_state["best_model_name"] is not None:
            st.success(f"推荐模型：{st.session_state['best_model_name']}")


# ==============================
# 数据导出
# ==============================
def render_export_page():
    render_system_title()
    st.subheader("数据导出")

    if st.session_state["df_processed"] is None:
        st.warning("请先完成数据预处理后再导出。")
        return

    df_to_export = st.session_state["df_processed"]
    csv_data = df_to_export.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="下载处理后的数据",
        data=csv_data,
        file_name="processed_data.csv",
        mime="text/csv"
    )

    st.success("当前导出内容为处理后的数据文件。")


# ==============================
# 主程序入口
# ==============================
if not st.session_state["logged_in"]:
    render_login()
else:
    page = render_sidebar()

    if page == "首页":
        render_home()
    elif page == "数据上传":
        render_upload_page()
    elif page == "原始数据展示":
        render_data_preview()
    elif page == "数据预处理":
        render_preprocess_page()
    elif page == "数据可视化":
        render_visualization_page()
    elif page == "模型训练":
        render_training_page()
    elif page == "数据导出":
        render_export_page()
