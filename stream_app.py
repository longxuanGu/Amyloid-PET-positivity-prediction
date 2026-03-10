import streamlit as st
import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import os
import streamlit.components.v1 as components

# ========================
# Global Config (SCI Style & 中文支持)
# ========================
st.set_page_config(
    page_title="淀粉样蛋白PET阳性预测",
    layout="centered"
)

# ⚠️ 终极修改：动态加载同目录下的字体文件，彻底解决跨平台 Matplotlib 中文乱码
font_path = "simhei.ttf"  # 确保此字体文件与 app.py 在同一目录下

if os.path.exists(font_path):
    # 将字体添加到 matplotlib 的字体管理器中
    fm.fontManager.addfont(font_path)
    # 获取字体的英文名称并设置为全局字体
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = prop.get_name()
else:
    st.warning("⚠️ 未在当前目录找到中文字体文件 `simhei.ttf`，图表中文可能显示为方块。")
    # Fallback 方案 (仅在本地 Windows 系统可能有救)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# ========================
# Load Model (RandomForest)
# ========================
@st.cache_resource
def load_model():
    # 请确保 rf_model.pkl 在同级目录下
    with open("rf_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ========================
# Title & Sidebar Inputs
# ========================
st.title("淀粉样蛋白PET阳性预测")

st.sidebar.header("输入特征")

cingulum_mid_r = st.sidebar.slider("右侧扣带回中部FDG SUVR", 0.0, 3.0, 0.0, 0.01)
cingulum_post_l = st.sidebar.slider("左侧扣带回后部FDG SUVR", 0.0, 3.0, 0.0, 0.01)
occipital_mid_l = st.sidebar.slider("左侧枕中回FDG SUVR", 0.0, 3.0, 0.0, 0.01)
precuneus_r = st.sidebar.slider("右侧楔前叶FDG SUVR", 0.0, 3.0, 0.0, 0.01)
temporal_mid_l = st.sidebar.slider("左侧颞中回FDG SUVR", 0.0, 3.0, 0.0, 0.01)

# ========================
# Construct Input Data
# ========================

# 1. 喂给机器学习模型的数据格式（必须严格保持英文特征名，防止模型报错）
input_data = pd.DataFrame({
    "Cingulum_Mid_R_FDG": [cingulum_mid_r],
    "Cingulum_Post_L_FDG": [cingulum_post_l],
    "Occipital_Mid_L_FDG": [occipital_mid_l],
    "Precuneus_R_FDG": [precuneus_r],
    "Temporal_Mid_L_FDG":[temporal_mid_l]
})

# 2. 中英文特征映射字典 (用于给 SHAP 注入中文标签)
feature_name_mapping = {
    "Cingulum_Mid_R_FDG": "右侧扣带回中部FDG SUVR",
    "Cingulum_Post_L_FDG": "左侧扣带回后部FDG SUVR",
    "Occipital_Mid_L_FDG": "左侧枕中回FDG SUVR",
    "Precuneus_R_FDG": "右侧楔前叶FDG SUVR",
    "Temporal_Mid_L_FDG": "左侧颞中回FDG SUVR"
}

# 提取按顺序排列的中文特征名列表
chinese_feature_names = [feature_name_mapping[col] for col in input_data.columns]


# ========================
# Prediction + SHAP
# ========================

if st.button("运行预测", type="primary", use_container_width=True):

    with st.spinner("正在计算预测结果与SHAP解释..."):

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
                1: "淀粉样蛋白PET预测阳性"
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
                # 处理部分模型 binary 预期值为数组的情况
                if isinstance(base, (list, np.ndarray)):
                    base = base[1] if len(base)>1 else base[0]

            # ⚠️ 关键修改：重构 Explanation 对象，强制填入 `chinese_feature_names` 替换原来的英文列名
            explanation = shap.Explanation(
                values=sv,
                base_values=base,
                data=input_data.iloc[0].values, # 传入数值本身
                feature_names=chinese_feature_names # 注入中文列名
            )

            # ========================
            # Panel A — Force Plot
            # ========================
            st.markdown("### **SHAP力图**")

            # 生成力图并强制传入中文特征名
            force_plot_html = shap.plots.force(
                explanation.base_values,
                explanation.values,
                explanation.data,
                feature_names=chinese_feature_names,
                matplotlib=False,
                show=False
            )
            
            shap.save_html("temp_force_plot.html", force_plot_html)  # 保存到临时文件
            with open("temp_force_plot.html", "r", encoding="utf-8") as f:
                components.html(f.read(), height=300, scrolling=True)
                
            # 清理临时文件
            if os.path.exists("temp_force_plot.html"):
                os.remove("temp_force_plot.html")

            st.caption("红色表示推动模型预测为阳性，蓝色表示推动模型预测为阴性")

            # ========================
            # Panel B — Waterfall
            # ========================
            st.markdown("### **SHAP瀑布图**")

            figB = plt.figure(figsize=(8, 5.5))
            
            # 这里的 explanation 已经自带中文名了，且 matplotlib 已加载中文字体，直接画图即可
            shap.plots.waterfall(explanation, show=False)
            
            st.pyplot(figB)
            plt.close(figB)

            st.caption("各脑区FDG代谢对淀粉样蛋白阳性预测的贡献")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

