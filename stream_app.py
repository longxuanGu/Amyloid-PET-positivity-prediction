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
    page_title="淀粉样蛋白PET阳性预测 | Amyloid PET Positivity Prediction",
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
st.subheader("Amyloid PET Positivity Prediction (Random Forest + SHAP)")

st.sidebar.header("输入特征 | Input Features")

cingulum_mid_r = st.sidebar.slider(
    "右侧中扣带回FDG SUVR | Cingulum_Mid_R_FDG", 0.0, 3.0, 0.0, 0.01
)
cingulum_post_l = st.sidebar.slider(
    "左侧后扣带回FDG SUVR | Cingulum_Post_L_FDG", 0.0, 3.0, 0.0, 0.01
)
occipital_mid_l = st.sidebar.slider(
    "左侧枕中回FDG SUVR | Occipital_Mid_L_FDG", 0.0, 3.0, 0.0, 0.01
)
precuneus_r = st.sidebar.slider(
    "右侧楔前叶FDG SUVR | Precuneus_R_FDG", 0.0, 3.0, 0.0, 0.01
)
temporal_mid_l = st.sidebar.slider(
    "左侧颞中回FDG SUVR | Temporal_Mid_L_FDG", 0.0, 3.0, 0.0, 0.01
)

# ========================

# Construct Input Data

# ========================

input_data = pd.DataFrame({
    "Cingulum_Mid_R_FDG": [cingulum_mid_r],
    "Cingulum_Post_L_FDG": [cingulum_post_l],
    "Occipital_Mid_L_FDG": [occipital_mid_l],
    "Precuneus_R_FDG": [precuneus_r],
    "Temporal_Mid_L_FDG": [temporal_mid_l]
})

# ========================

# Prediction + SHAP

# ========================

if st.button("运行预测 | Run Prediction", type="primary", use_container_width=True):

    with st.spinner("正在计算预测结果与SHAP解释 | Generating prediction and SHAP explanations..."):

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
                0: "淀粉样蛋白PET预测阴性 | Amyloid PET Negative",
                1: "淀粉样蛋白PET预测阳性 | Amyloid PET Positive"
            }

            st.success(f"**预测结果 | Predicted Class:** {class_map.get(pred_class, str(pred_class))}")

            if prob_pos is not None:
                st.write(f"**阳性概率 | Probability of Amyloid Positivity:** {prob_pos:.3f}")
            else:
                st.write("**阳性概率 | Probability:** N/A")

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
            st.markdown("### **SHAP力图 | SHAP Force Plot**")

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
                "红色表示推动模型预测为阳性，蓝色表示推动模型预测为阴性 | "
                "Red pushes toward positive prediction, blue toward negative"
            )

            # ========================
            # Panel B — Waterfall
            # ========================
            st.markdown("### **SHAP瀑布图 | SHAP Waterfall Plot**")

            figB = plt.figure(figsize=(8, 5.5))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(figB)
            plt.close(figB)

            st.caption(
                "各脑区FDG代谢对淀粉样蛋白阳性预测的贡献 | "
                "Contribution of each brain region to amyloid positivity prediction"
            )

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# ========================

# Footer

# ========================

st.markdown("---")
st.caption("淀粉样蛋白PET阳性预测 | Amyloid PET Positivity Prediction App © 2026")