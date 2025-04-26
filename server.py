from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.optimizers import Adam
import base64
import math
import time

# ─────────────────────────────────────────────
# Custom layer override for TensorFlow MobileNet
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# ─────────────────────────────────────────────
# Load Model and Labels
model = load_model(
    'model/keras_model.h5',
    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

with open("model/labels.txt", "r") as f:
    labels = [line.strip().split(' ')[1] for line in f]

# ─────────────────────────────────────────────
# App Setup
app = Flask(__name__)
CORS(app)  # Allow CORS for React frontend

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# ─────────────────────────────────────────────
# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode image
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        hands, img = detector.findHands(img)
        if not hands:
            return jsonify({'prediction': 'No hand detected'})

        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop hand with offset
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            return jsonify({'prediction': 'Invalid hand crop'})

        # Create white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Preprocess and predict
        imgFinal = cv2.resize(imgWhite, (224, 224))
        imgFinal = imgFinal.astype('float32') / 255.0
        prediction = model.predict(imgFinal[np.newaxis, ...])[0]
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        label = labels[index]

        duration = round((time.time() - start_time) * 1000, 2)  # ms

        print(f"[INFO] Predicted: {label} | Confidence: {confidence:.2f} | Time: {duration}ms")

        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 2),
            'time_ms': duration
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
