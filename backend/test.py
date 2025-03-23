import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Load the trained model
model = tf.keras.models.load_model('deepfake_detector.h5')

# Image size (same as training)
img_size = 128

def predict_image(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize
        
        # Make prediction
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
        
        if confidence > 0.5:
            print(f"Prediction: FAKE ({confidence*100:.2f}% confidence)")
        else:
            print(f"Prediction: REAL ({(1-confidence)*100:.2f}% confidence)")
    
    except Exception as e:
        print(f"Error processing image: {e}")

# Usage: python test_model.py path_to_image.jpg
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py path_to_image.jpg")
    else:
        img_path = sys.argv[1]
        predict_image(img_path)
