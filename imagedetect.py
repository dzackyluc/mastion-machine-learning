import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model

# Load the model (ensure you have TensorFlow/Keras installed)
model = load_model('mastion_cnn_kaggle_models_trained.h5')

model.compile(optimizer='adam', loss='binary_crossentropy')

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model's expected input dimensions
    img = img / 255.0  # Normalize pixel values if model expects it
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Process an image and predict
image_path = 'datasets_2/test/0/Picture50-400x284.jpg'  # Path to your image, e.g., 'path/to/your/
processed_image = preprocess_image(image_path)
predictions = model.predict(processed_image)

# Process predictions (this step depends on your model's output)
# For example, if it's a classification model:
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class}")

# Note: This is a very basic example. Your actual implementation might need adjustments based on your model's architecture and requirements.