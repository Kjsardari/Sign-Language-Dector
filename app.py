import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.optimizers import Adam

# Custom DepthwiseConv2D that ignores the 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  #remove the parameter taht a
        super().__init__(*args, **kwargs)

# Custom Classifier
class CustomClassifier:
    def __init__(self, modelPath, labelsPath):
        self.model_path = modelPath
        self.labels_path = labelsPath
        self.model = None
        self.labels = []
        self.load_model()
        self.load_labels()
        
    def load_model(self):
        self.model = load_model(
            self.model_path,
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
        )
        # Compile the model if needed
        if not hasattr(self.model, 'optimizer'):
            self.model.compile(optimizer=Adam(), 
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
        
    def load_labels(self):
        with open(self.labels_path, 'r') as f:
            self.labels = f.read().splitlines()
            
    def getPrediction(self, img, draw=True):
        # Resize to model's expected input shape (224x224)
        img_resized = cv2.resize(img, (224, 224))
        # Normalize if needed (assuming model expects values 0-1)
        img_normalized = img_resized.astype('float32') / 255.0
        prediction = self.model.predict(img_normalized[np.newaxis, ...])
        index = np.argmax(prediction)
        return prediction, index

# Main code
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = CustomClassifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 300  # For cropping, but model expects 224x224
labels = ["A","B","C","D","DELETE","F","N","SPACE","K"]

while True:
    success, img = cap.read()
    if not success:
        continue
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure the crop coordinates are within image bounds
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        
        imgCrop = img[y1:y2, x1:x2]
        
        # Skip if crop is empty
        if imgCrop.size == 0:
            continue
            
        # Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        aspectRatio = h / w
        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                
            # Get prediction (will be resized to 224x224 inside getPrediction)
            prediction, index = classifier.getPrediction(imgWhite)
            
            # Display results
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                         (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, classifier.labels[index], (x, y -26), 
                       cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                         (x + w+offset, y + h+offset), (255, 0, 255), 4)
            
        except Exception as e:
            print(f"Error processing hand: {str(e)}")
            continue
            
    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()