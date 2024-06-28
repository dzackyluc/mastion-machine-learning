import cv2
import h5py
import numpy as np
from keras._tf_keras.keras.models import load_model

# Load the model
model = load_model("mastion_cnn_models_trained.h5")

# Initialize the camera
camera = cv2.VideoCapture(2)  # 0 represents the default camera

while True:
    # Read a frame from the camera
    ret, frame = camera.read()
    if not ret:
        break  # If the frame is not captured successfully, exit the loop

    # Convert the frame to RGB and resize it to match the model's expected input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (224, 224))  # Example size, adjust to your model's input

    # Normalize and expand dimensions to match the model's expected input
    frame_normalized = resized_frame / 255.0
    frame_batch = np.expand_dims(frame_normalized, axis=0)

    # Make a prediction
    predictions = model.predict(frame_batch)

    # Process the predictions here
    # For example, for a classification model, find the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)

    # Display the prediction on the frame
    cv2.putText(frame, f"Class: {predicted_class}, Confidence: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord("q"):  # Press 'q' to exit
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()