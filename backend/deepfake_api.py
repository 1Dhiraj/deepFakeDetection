from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import uvicorn

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, change "*" to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = tf.keras.models.load_model("deepfake_detector.h5", compile=False)

# Prediction Function
def predict_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            result = {"prediction": "REAL", "confidence": f"{confidence * 100:.2f}%"}
        else:
            result = {"prediction": "FAKE", "confidence": f"{(1 - confidence) * 100:.2f}%"}


        return result
    except Exception as e:
        return {"error": str(e)}

# API Route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = predict_image(img_bytes)
    return JSONResponse(content=result)

# Run Server
if __name__ == "__main__":
    uvicorn.run("deepfake_api:app", host="0.0.0.0", port=8000, reload=True)
