from keras._tf_keras.keras.models import load_model
import numpy as np
import cv2

# Membuat model
classifier = load_model('mastion_cnn_models_trained.h5')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # 0 menandakan kamera utama

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()

    # Preprocess frame untuk object detection
    frame = cv2.resize(frame, (224, 224))
    frame = frame.reshape((1, 224, 224, 3))
    frame = frame.astype(np.float32) / 255.0

    # Melakukan prediksi menggunakan model
    prediction = classifier.predict(frame)

    # Mendapatkan label dari prediksi
    label = np.argmax(prediction)

    # Menampilkan hasil prediksi dengan bounding box dan label
    if label == 0:
        # Mendapatkan koordinat bounding box
        x, y, w, h = 10, 10, 100, 100  # Define the initial values of x, y, w, h

        # Menggambar bounding box dan label pada frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, 'Terkena Mastitis', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if label == 1:
        # Mendapatkan koordinat bounding box
        x, y, w, h = 20, 20, 200, 200  # Define the initial values of x, y, w, h

        # Menggambar bounding box dan label pada frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Tidak Terkena Mastitis', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Debugging: Print frame shape and type before displaying
    if frame is not None and len(frame.shape) == 2:
        height, width, channels = frame.shape
        print("Frame Shape:", frame.shape)
        print("Frame Type:", frame.dtype)

        # Validate frame dimensions and type
        if height > 0 and width > 0 and (channels == 1 or channels == 3 or channels == 4) and frame.dtype == 'uint8':
            # Menampilkan frame dengan hasil prediksi
            cv2.imshow('Object Detection', frame)
        else:
            print("Invalid frame dimensions or type.")
    else:
        print("Frame is empty or not in expected format.")

    # Tombol q untuk keluar dari program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
