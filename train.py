import os
import numpy as np
import keras._tf_keras as tf
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img
from keras._tf_keras.keras.utils import Sequence

# menset path dataset train dan test
train_data_dir = './datasets_2/train'
test_data_dir = './datasets_2/test'

# menset jumlah kelas, ukuran gambar, dan batch size
num_classes = 1
img_width, img_height = 224, 224
batch_size = 32

# membuat custom dataset
class CustomDataset(Sequence):
    def __init__(self, data_dir, batch_size, img_width, img_height):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.image_paths = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for class_label in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_label)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.image_paths.append(image_path)
                self.labels.append(int(class_label))

    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_x = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([img_to_array(load_img(file, target_size=(self.img_width, self.img_height))) for file in batch_x]), np.array(batch_y)

# Membuat objek dataset train dan test
train_generator = CustomDataset(train_data_dir, batch_size, img_width, img_height)
test_generator = CustomDataset(test_data_dir, batch_size, img_width, img_height)

# Build your model using TensorFlow's Sequential API
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
])

print(model.summary())

#Rate ketika model mencapai titik tertentu akan dinonaktifkan
class ThressholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThressholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["accuracy"]
        if accuracy > self.threshold:
            self.model.stop_training = True

end_train = ThressholdCallback(threshold=0.90)

# mengkompilasi model
model.compile(optimizer='adam',
              loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
              metrics=['accuracy'])

# Melatih model
model.fit(train_generator,
          #steps_per_epoch=train_generator.samples // batch_size,
          epochs=50,
          validation_data=test_generator,
          callbacks=[end_train],
          #validation_steps=test_generator.samples // batch_size,
          )

model.save('mastion_cnn_kaggle_models_trained.h5')