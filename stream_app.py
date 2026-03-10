import streamlit as st
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ========================

# Global Config (SCI Style)

# ========================

st.set_page_config(
    page_title="淀粉样蛋白PET阳性预测",
    layout="centered"
)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# ========================

# Load Model (RandomForest)

# ========================

@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ========================

# Title & Sidebar Inputs

# ========================

st.title("淀粉样蛋白PET阳性预测")

st.sidebar.header("输入特征 ")

cingulum_mid_r = st.sidebar.slider(
    "右侧扣带回中部FDG SUVR", 0.0, 3.0, 0.0, 0.01
)
cingulum_post_l = st.sidebar.slider(
    "左侧扣带回后部FDG SUVR", 0.0, 3.0, 0.0, 0.01
)
occipital_mid_l = st.sidebar.slider(
    "左侧枕中回FDG SUVR", 0.0, 3.0, 0.0, 0.01
)
precuneus_r = st.sidebar.slider(
    "右侧楔前叶FDG SUVR", 0.0, 3.0, 0.0, 0.01
)
temporal_mid_l = st.sidebar.slider(
    "左侧颞中回FDG SUVR", 0.0, 3.0, 0.0, 0.01
)

# ========================

# Construct Input Data

# ========================

# 1. 构建用于模型预测的 DataFrame（必须与训练集的特征名保持完全一致！）
input_data_model = pd.DataFrame({
    "Cingulum_Mid_R_FDG": [cingulum_mid_r],
    "Cingulum_Post_L_FDG": [cingulum_post_l],
    "Occipital_Mid_L_FDG": [occipital_mid_l],
    "Precuneus_R_FDG": [precuneus_r],
    "Temporal_Mid_L_FDG":[temporal_mid_l]
})

# 执行预测 (使用英文列名的数据)
# prediction = model.predict(input_data_model) 

# ---------------------------------------------------------

# 2. 构建用于 Streamlit 网页表格展示 / SHAP 可视化的 DataFrame
# 使用 rename 方法将英文列名映射为中文
feature_name_mapping = {
    "Cingulum_Mid_R_FDG": "右侧扣带回中部FDG SUVR",
    "Cingulum_Post_L_FDG": "左侧扣带回后部FDG SUVR",
    "Occipital_Mid_L_FDG": "左侧枕中回FDG SUVR",
    "Precuneus_R_FDG": "右侧楔前叶FDG SUVR",
    "Temporal_Mid_L_FDG": "左侧颞中回FDG SUVR"
}

input_data_display = input_data_model.rename(columns=feature_name_mapping)

# 在网页上展示中文表格
# st.dataframe(input_data_display) 

# ========================

# Prediction + SHAP

# ========================

if st.button("运行预测", type="primary", use_container_width=True):

    with st.spinner("正在计算预测结果与SHAP解释"):

        try:
            # ---------- Prediction ----------
            pred_class = int(model.predict(input_data)[0])

            prob_pos = None
            try:
                proba = model.predict_proba(input_data)[0]
                prob_pos = float(proba[1])
            except Exception:
                prob_pos = None

            class_map = {
                0: "淀粉样蛋白PET预测阴性",
                1: "淀粉样蛋白PET预测阳性 "
            }

            st.success(f"**预测结果:** {class_map.get(pred_class, str(pred_class))}")

            if prob_pos is not None:
                st.write(f"**阳性概率:** {prob_pos:.3f}")
            else:
                st.write("**阳性概率:** N/A")

            # ---------- SHAP ----------
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_data)

            if shap_values.values.ndim == 3:  # multiclass
                sv = shap_values.values[0, :, pred_class]
                base = explainer.expected_value[pred_class]
            else:
                sv = shap_values.values[0]
                base = explainer.expected_value

            explanation = shap.Explanation(
                values=sv,
                base_values=base,
                data=input_data.iloc[0],
                feature_names=input_data.columns
            )

            # ========================
            # Panel A — Force Plot
            # ========================
            st.markdown("### **SHAP力图**")

            force_plot_html = shap.plots.force(
                explanation,
                matplotlib=False,  # 默认 JS/HTML 模式
                show=False
            )
            shap.save_html("temp_force_plot.html", force_plot_html)  # 保存到临时文件
            with open("temp_force_plot.html", "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=300, scrolling=True)
            # 清理临时文件
            if os.path.exists("temp_force_plot.html"):
                os.remove("temp_force_plot.html")

            st.caption(
                "红色表示推动模型预测为阳性，蓝色表示推动模型预测为阴性 ")

            # ========================
            # Panel B — Waterfall
            # ========================
            st.markdown("### **SHAP瀑布图**")

            figB = plt.figure(figsize=(8, 5.5))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(figB)
            plt.close(figB)

            st.caption(
                "各脑区FDG代谢对淀粉样蛋白阳性预测的贡献")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

