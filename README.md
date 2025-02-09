# **Face Mask Detection Using Deep Learning**

## **Executive Summary**
The COVID-19 pandemic has emphasized the importance of face masks in preventing the spread of infectious diseases. As a Data Scientist at *SafeGuard AI Solutions*, my role is to develop an AI-powered face mask detection system using deep learning techniques. This system will automate mask detection in public spaces, helping businesses, security agencies, and healthcare institutions enforce mask policies effectively.  

By leveraging Convolutional Neural Networks (CNNs), this project aims to create a robust model capable of accurately classifying individuals as wearing a mask or not. The ultimate goal is to deploy this system for real-time surveillance in high-traffic areas, ensuring public health compliance and safety.  

## **Project Objective**
- **Develop a deep learning model** to classify individuals as wearing or not wearing a face mask.
- **Train and evaluate** a CNN-based model for high accuracy in mask detection.
- **Optimize model performance** to ensure real-time usability in security and surveillance applications.
- **Deploy the model** for integration into security camera systems, mobile applications, or web interfaces.
- **Improve public safety** by supporting enforcement of mask-wearing policies in public spaces.

## **Data Collection**
### **1. Dataset Source**
The dataset used for this project is obtained from Kaggle's *Face Mask Dataset* ([link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)). It contains images of individuals with and without masks, which will be used for training and evaluation.

### **2. Data Categories**
The dataset consists of:
- **With Mask:** Images of people correctly wearing face masks.
- **Without Mask:** Images of people not wearing face masks.

### **3. Data Preprocessing**
- Resizing all images to **128x128 pixels** for consistency.
- Normalizing pixel values to a **0-1 range** for better model performance.
- Converting images to **RGB format** to ensure compatibility with deep learning frameworks.
- Splitting the dataset into **80% training and 20% testing** for evaluation.


---

## Exploratory Data Analysis (EDA)
EDA helped me to understand the dataset structure, distribution, and sample images before training the model.

### 1. Sample Images
To visualize sample images from the dataset:

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display a sample image from each category
img = mpimg.imread("/content/data/with_mask/with_mask_538.jpg")
imgplot = plt.imshow(img)
plt.show()
```

![Face mask](https://github.com/user-attachments/assets/b2b06f56-2e80-4a4a-b7d7-0f82043aaac8)

```python
img = mpimg.imread("/content/data/without_mask/without_mask_711.jpg")
imgplot = plt.imshow(img)
plt.show()
```

![Face mask 2](https://github.com/user-attachments/assets/baeac4bb-5c5c-4166-a881-59908b3e93ae)


##  2. Data Preprocessing
###  Labeling Data
- **1** represents "With Mask"  
- **0** represents "Without Mask"

```python
# Assign labels
with_mask_labels = [1] * len(with_mask_files)
without_mask_labels = [0] * len(without_mask_files)

# Combine labels
labels = with_mask_labels + without_mask_labels
```
---

## Model Development

### 1.1 Building a CNN Model
A **Convolutional Neural Network (CNN)** was used to classify images based on patterns in pixels.

```python
import tensorflow as tf
from tensorflow import keras

# Define CNN architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Display model summary
model.summary()
```

---

## 1.2 Model Training
I trained the model on the **training dataset** while validating on a small portion.

```python
# Train model
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=10, batch_size=32)
```

---

## Model Evaluation
### Evaluating Test Accuracy
```python
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Performance Visualization
#### Training & Validation Loss
```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()
```
![Training Loss](https://github.com/user-attachments/assets/80855e88-4155-41c9-8aa5-f4b2b930766b)


#### Training & Validation Accuracy
```python
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.show()
```
![Training Accuracy](https://github.com/user-attachments/assets/da847369-5210-4c6c-9fe8-2529a7554c6e)

---

## Face Mask Prediction
To make predictions on new images:

```python
import cv2

def predict_mask(image_path):
    input_image = cv2.imread(image_path)
    input_image_resized = cv2.resize(input_image, (128,128))
    input_image_scaled = input_image_resized / 255.0
    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

    prediction = model.predict(input_image_reshaped)
    predicted_label = np.argmax(prediction)

    if predicted_label == 1:
        print("✅ The person in the image is wearing a mask")
    else:
        print("❌ The person in the image is NOT wearing a mask")


```

---

## Findings
- The CNN model achieved **high accuracy** of **92%** in detecting face masks.
- The dataset was **well-balanced**, aiding in better generalization.
- Training on **more diverse datasets** could further improve results.

---

## Recommendations
- **Improve Dataset:** Make use of real-world images to reduce biases.
- **Fine-Tune Hyperparameters:** Experiment with different architectures and learning rates.
- **Deploy Model:** Convert the trained model into a web or mobile app.

---

## Limitations
- Performance may degrade on **unseen variations** (e.g., low lighting, occluded faces).
- **Overfitting** might occur due to limited dataset diversity.

---

## Future Work
- Implement real-time **face detection** before classification.
- Optimize the model for **faster inference** on edge devices.
- Develop a **mobile app** using TensorFlow Lite.

---
