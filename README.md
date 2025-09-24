# MNIST Digit Recognizer

A web application that allows users to draw handwritten digits (0–9) and predicts the digit using a **PyTorch-trained MNIST model exported to ONNX**, served via **FastAPI**, with a **Streamlit frontend** for real-time interaction.

---

## Features

* Draw digits on a canvas with real-time predictions.
* Backend powered by **ONNX Runtime** for fast inference.
* Streamlit frontend shows predicted class and class probabilities.
* Supports Docker Compose for easy deployment.

---

## Directory Structure

```
. 
├── services 
│   ├── airflow/            # All data related to airflow DAGs                              
├── dataset/               # Raw MNIST dataset files
├── models/                # Trained models (ONNX, PyTorch) and metrics
├── src/
│   ├── frontend/          # Streamlit app
│   └── ml/                # FastAPI backend and model training code
├── docker-compose.yml     # Docker Compose setup
├── requirements.txt
└── README.md
```

---

## Requirements

* Docker & Docker Compose
* Python 3.9 or neweer

## Setup & Running

### 1. Clone the repository

```bash
git clone https://github.com/LowIQCoder/PMLDL_A1
cd PMLDL_A1
```

### 2. Set up the envoirement

First of all prepare env for aitflow

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
export AIRFLOW_HOME="$PWD/services/airflow"
export AIRFLOW__API__WORKERS=1
```

And install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install "apache-airflow==2.9.3" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.11.txt"
pip install -r requirements.txt
```

### 3. Start DAG

Very simple
```bash
airflow standalone
```

* **Airflow**: `http://localhost:8080`
* **MLFlow**: `htt[://localhost:5000`
* **Backend (FastAPI)**: `http://localhost:8000/docs`
* **Frontend (Streamlit)**: `http://localhost:8501`

## Usage

1. Open the Streamlit app (`http://localhost:8501`).
2. Draw a digit (0–9) on the canvas.
3. Predictions are updated **in real-time**.
4. The app displays:

   * **Predicted class**
   * **Class probabilities** as a bar chart

---

## Model

* The model is trained on the **MNIST dataset** using **PyTorch**.
* Exported to **ONNX format** for fast inference in the backend.
* Expected input: 28×28 grayscale image, white digit on black background, normalized to \[0,1].

---

## License


