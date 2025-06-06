import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import joblib

# --- 預載模型 ---
model = joblib.load("model/trained_model.pkl")

# --- 標題區 ---
st.title("PM2.5 ROI 分析與預測系統 🌫️")
st.markdown("上傳影像後，自動進行差異區域偵測與超標預測。")

# --- 圖片上傳 ---
uploaded_file = st.file_uploader("📤 請上傳處理後的差異影像 (binary 或 diff)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="你上傳的影像", use_column_width=True)

    # TODO: 加上自動處理 ROI → 特徵抽取
    st.warning("🚧 尚未自動處理影像 ROI，請使用已處理的 CSV 先預測")

# --- CSV 預測結果展示 ---
st.header("📋 預測結果總覽")

csv_path = "data/ROI預測結果.csv"
try:
    df = pd.read_csv(csv_path)
    st.dataframe(df[["diff_index", "total_area", "label", "predicted_label", "prob_exceed"]])

    # 統計資訊
    st.metric("總筆數", len(df))
    st.metric("預測超標數量", (df["predicted_label"] == 1).sum())

    # 機率分布圖
    st.subheader("📈 機率分布直方圖")
    fig, ax = plt.subplots()
    df[df["label"] == 1]["prob_exceed"].plot.hist(bins=20, alpha=0.5, color="red", label="超標", ax=ax)
    df[df["label"] == 0]["prob_exceed"].plot.hist(bins=20, alpha=0.5, color="green", label="合格", ax=ax)
    plt.axvline(x=0.5, color="blue", linestyle="--", label="閾值 = 0.5")
    plt.xlabel("超標機率")
    plt.ylabel("頻率")
    plt.legend()
    st.pyplot(fig)

    # 散點圖
    st.subheader("🔍 ROI 面積 vs 預測機率")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df[df["label"] == 1]["total_area"], df[df["label"] == 1]["prob_exceed"], color="red", alpha=0.5, label="超標")
    ax2.scatter(df[df["label"] == 0]["total_area"], df[df["label"] == 0]["prob_exceed"], color="green", alpha=0.5, label="合格")
    ax2.axhline(0.5, color="blue", linestyle="--")
    ax2.set_xlabel("ROI 面積")
    ax2.set_ylabel("預測超標機率")
    ax2.legend()
    st.pyplot(fig2)

except FileNotFoundError:
    st.error("❌ 找不到 ROI預測結果.csv，請先執行 Python 預測")


