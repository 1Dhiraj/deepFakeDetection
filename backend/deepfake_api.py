from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
import uvicorn

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
model = tf.keras.models.load_model("deepfake_detector.h5", compile=False)

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(img: Image.Image) -> Image.Image:
    # Convert PIL image to grayscale OpenCV image
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    
    # Assuming the first detected face is the target
    x, y, w, h = faces[0]
    face_img = cv_img[y:y+h, x:x+w]
    
    # Convert back to PIL image
    face_pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return face_pil_img

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Detect and crop the face
        face_img = detect_and_crop_face(img)
        
        # Preprocess the cropped face image
        preprocessed_img = preprocess_image(face_img)
        
        # Predict using the model
        prediction = model.predict(preprocessed_img)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            result = {"prediction": "REAL", "confidence": f"{confidence * 100:.2f}%"}
        else:
            result = {"prediction": "FAKE", "confidence": f"{(1 - confidence) * 100:.2f}%"}
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Run the server
if __name__ == "__main__":
    uvicorn.run("deepfake_api:app", host="0.0.0.0", port=8000, reload=True)
