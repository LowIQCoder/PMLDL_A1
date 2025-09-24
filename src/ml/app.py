from fastapi import FastAPI, UploadFile, File
import onnxruntime as rt
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI()

session = rt.InferenceSession("models/model.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.get("/ping")
def pong():
    return {"message": "pong"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")

    image = image.resize((28, 28))
    img_array = np.array(image).astype(np.float32) / 255.0

    img_array = 1.0 - img_array

    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=0)

    preds = session.run([output_name], {input_name: img_array})[0][0]

    exp_preds = np.exp(preds - np.max(preds))
    probs = exp_preds / exp_preds.sum()

    results = [{"label": str(i), "probability": float(p)} for i, p in enumerate(probs)]
    predicted_class = int(np.argmax(probs))

    return {"prediction": predicted_class, "probabilities": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
