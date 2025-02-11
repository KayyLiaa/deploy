import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import matplotlib.pyplot as plt

# Path dataset
data_dir = os.path.abspath("dataset_daun-kentang")

# Pastikan dataset ada
dataset_exists = os.path.exists(data_dir) and any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir))
if not dataset_exists:
    raise FileNotFoundError("Dataset folder tidak ditemukan atau kosong! Pastikan folder dataset berisi gambar dalam subfolder kelas.")

# Cek isi dataset
print("Memeriksa dataset...")
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        if num_images == 0:
            raise ValueError(f"Folder kelas '{class_name}' tidak memiliki gambar! Tambahkan gambar untuk melanjutkan.")
        print(f"Kelas '{class_name}': {num_images} gambar")

# Image Data Generator dengan Augmentasi
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Meningkatkan split validation agar tidak kosong
    rotation_range=20,
    horizontal_flip=True
)

# Gunakan batch kecil agar dataset tidak kosong
batch_size = 4

# Load dataset
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42  # Menambahkan seed agar konsisten
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42  # Menambahkan seed agar konsisten
)

# Cek apakah data berhasil dimuat
print("Total batch training:", len(train_data))
print("Total batch validation:", len(val_data))

if len(train_data) == 0 or len(val_data) == 0:
    raise ValueError("Dataset tidak memiliki cukup gambar untuk training atau validation. Tambahkan lebih banyak gambar.")

# Simpan mapping label kelas ke JSON
class_indices = train_data.class_indices
with open("class_labels.json", "w") as f:
    json.dump(class_indices, f)

# Model CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_indices), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25
)

# Simpan model
model.save("daun_kentang.h5")
print("Model dan label kelas telah disimpan!")

# Plot Grafik Akurasi & Loss
plt.figure(figsize=(12, 5))

# Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Simpan grafik
save_dir = "static/images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

graph_path = os.path.join(save_dir, "training_results.png")
plt.savefig(graph_path, dpi=300)
plt.show()
