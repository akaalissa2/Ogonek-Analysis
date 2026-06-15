import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(
    page_title="Анализ изображения",
    layout="wide"
)

st.title("Анализ изображения")

st.sidebar.header("Настройки")
n_colors = st.sidebar.slider("Число определяемых цветов", 2, 8, 3)
detect_faces_flag = st.sidebar.checkbox("Определение лиц", True)
red_analysis_flag = st.sidebar.checkbox("Анализ красного цвета", True)

@st.cache_data
def red_ratio_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)
    return (np.sum(mask > 0) / mask.size) * 100


@st.cache_data
def detect_faces(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5)
    return len(faces)


@st.cache_data
def dominant_colors(img_rgb, n_colors=3):
    pixels = img_rgb.reshape((-1, 3))

    model = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    model.fit(pixels)

    colors = model.cluster_centers_.astype(int)
    labels, counts = np.unique(model.labels_, return_counts=True)
    percents = counts / counts.sum() * 100

    order = np.argsort(percents)[::-1]
    return colors[order], percents[order]


def color_chart(colors, percents):
    df = pd.DataFrame({
        "color": [str(tuple(c)) for c in colors],
        "percent": percents
    })

    fig = px.pie(df, names="color", values="percent")
    return fig

uploaded_file = st.file_uploader("Загрузить изображение", type=["jpg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_rgb, use_container_width=True)

    with col2:

        if red_analysis_flag:
            red = red_ratio_hsv(img_bgr)
            st.metric("Red ratio", f"{red:.2f}%")

        if detect_faces_flag:
            faces = detect_faces(img_bgr)
            st.metric("Лица", faces)

    st.divider()

    st.subheader("Главные цвета")

    colors, percents = dominant_colors(img_rgb, n_colors)

    colA, colB = st.columns(2)

    with colA:
        df = pd.DataFrame({
            "color": [str(tuple(c)) for c in colors],
            "percent": percents
        })
        st.dataframe(df, hide_index=True)

    with colB:
        st.plotly_chart(color_chart(colors, percents), use_container_width=True)

    st.download_button(
        "Скачать CSV",
        pd.DataFrame({
            "color": [str(tuple(c)) for c in colors],
            "percent": percents
        }).to_csv(index=False),
        "report.csv",
        "text/csv"
    )