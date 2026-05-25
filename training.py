import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================================
# JUDUL PROGRAM
# ============================================================

print("="*60)
print("KLASIFIKASI CITRA WILAYAH PERKOTAAN")
print("MENGGUNAKAN CNN EFFICIENTNETB2")
print("="*60)

# =========================
# PREPROCESSING & AUGMENTASI
# =========================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2
)

# =========================
# DATA TRAINING
# =========================

train_generator = train_datagen.flow_from_directory(

    'dataset',

    target_size=(224,224),

    batch_size=16,

    class_mode='categorical',

    subset='training',

    shuffle=True
)

# =========================
# DATA VALIDATION
# =========================

val_generator = val_datagen.flow_from_directory(

    'dataset',

    target_size=(224,224),

    batch_size=16,

    class_mode='categorical',

    subset='validation',

    shuffle=False
)

# =========================
# MODEL EFFICIENTNETB2
# =========================

base_model = EfficientNetB2(

    weights='imagenet',

    include_top=False,

    input_shape=(224,224,3)
)

# =========================
# FINE TUNING
# =========================

base_model.trainable = True

for layer in base_model.layers[:-20]:

    layer.trainable = False

# =========================
# LAYER TAMBAHAN
# =========================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.3)(x)

output = Dense(3, activation='softmax')(x)

# =========================
# MODEL AKHIR
# =========================

model = Model(

    inputs=base_model.input,

    outputs=output
)

# =========================
# COMPILE MODEL
# =========================

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss='categorical_crossentropy',

    metrics=['accuracy']
)

# =========================
# CALLBACK
# =========================

callbacks = [


    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

]

# =========================
# TRAINING MODEL
# =========================

history = model.fit(

    train_generator,

    validation_data=val_generator,

    epochs=5,

    callbacks=callbacks
)

# =========================
# SIMPAN MODEL
# =========================

model.save("best_model.h5")

# =========================
# GRAFIK ACCURACY
# =========================

plt.figure(figsize=(8,5))

plt.plot(

    history.history['accuracy'],

    label='Training Accuracy'
)

plt.plot(

    history.history['val_accuracy'],

    label='Validation Accuracy'
)

plt.title('Grafik Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.savefig('accuracy.png')

# =========================
# GRAFIK LOSS
# =========================

plt.figure(figsize=(8,5))

plt.plot(

    history.history['loss'],

    label='Training Loss'
)

plt.plot(

    history.history['val_loss'],

    label='Validation Loss'
)

plt.title('Grafik Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.savefig('loss.png')

# =========================
# PREDIKSI VALIDATION
# =========================

val_generator.reset()

pred = model.predict(val_generator)

pred_classes = np.argmax(

    pred,

    axis=1
)

true_classes = val_generator.classes

class_labels = list(

    train_generator.class_indices.keys()
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(

    true_classes,

    pred_classes
)

print("\n========== CONFUSION MATRIX ==========\n")

print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=class_labels,

    yticklabels=class_labels
)

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

plt.savefig('confusion_matrix.png')

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(

    classification_report(

        true_classes,

        pred_classes,

        target_names=class_labels
    )
)

print("\n========== TRAINING SELESAI ==========")

print("Model berhasil disimpan")