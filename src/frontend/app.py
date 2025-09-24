import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import requests
import io

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("MNIST Digit Recognizer")
st.write("Draw a digit (0-9). Predictions update in real-time.")

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("L")
    img = img.resize((28, 28))

    img = ImageOps.invert(img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    try:
        response = requests.post(
            "http://backend:8000/predict",
            files={"file": buf}
        )
        response.raise_for_status()
        data = response.json()

        st.write(f"### Predicted Class: {data['prediction']}")

        probs = {item['label']: item['probability'] for item in data['probabilities']}
        st.bar_chart(probs)

    except requests.exceptions.RequestException as e:
        st.error(f"Prediction failed: {e}")
