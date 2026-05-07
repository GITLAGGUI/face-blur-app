import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Paths
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "faces.csv")

# Hyperparameters
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10  # Can be adjusted based on needs

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# To speed up training for the assignment, we can use a subset (e.g., 2000 images)
# For the full dataset, remove the .head()
df = df.head(3000)

images = []
targets = []

for index, row in df.iterrows():
    img_name = row['image_name']
    img_path = os.path.join(IMAGES_DIR, img_name)
    
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    # Resize image
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize image
    img_normalized = img_resized / 255.0
    
    images.append(img_normalized)
    
    # Normalize bounding box coordinates
    orig_w = row['width']
    orig_h = row['height']
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    
    norm_x0 = x0 / orig_w
    norm_y0 = y0 / orig_h
    norm_x1 = x1 / orig_w
    norm_y1 = y1 / orig_h
    
    targets.append([norm_x0, norm_y0, norm_x1, norm_y1])

X = np.array(images, dtype=np.float32)
y = np.array(targets, dtype=np.float32)

print(f"Loaded {len(X)} images.")

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Design the Custom CNN for Bounding Box Regression
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(4, activation='sigmoid')  # 4 coordinates (x0, y0, x1, y1) normalized between 0 and 1
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train the model
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Evaluate performance
loss, mae = model.evaluate(X_val, y_val)
print(f"Validation Loss (MSE): {loss:.4f}")
print(f"Validation MAE: {mae:.4f}")

# Save the model
model.save("custom_cnn_face_detection.h5")
print("Model saved to custom_cnn_face_detection.h5")
