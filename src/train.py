# ============================================
# Animal Breed Classification - InceptionV3
# Training Script
# ============================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ============================================
# PARAMETERS
# ============================================

DATASET_PATH = "/content/drive/MyDrive/dataset1/train"

IMG_SIZE = 299
BATCH_SIZE = 32

PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20

# ============================================
# DATA AUGMENTATION
# ============================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# ============================================
# DATA GENERATORS
# ============================================

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

NUM_CLASSES = train_generator.num_classes

print("Number of classes:", NUM_CLASSES)

# ============================================
# LOAD BASE MODEL
# ============================================

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# ============================================
# FREEZE BASE MODEL
# ============================================

for layer in base_model.layers:
    layer.trainable = False

# ============================================
# ADD CUSTOM CLASSIFICATION HEAD
# ============================================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)

predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ============================================
# COMPILE MODEL (PHASE 1)
# ============================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# TRAIN PHASE 1
# ============================================

history_phase1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=PHASE1_EPOCHS
)

# ============================================
# FINE TUNING (UNFREEZE LAST 100 LAYERS)
# ============================================

for layer in base_model.layers[-100:]:
    layer.trainable = True

# ============================================
# RECOMPILE MODEL
# ============================================

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# CALLBACKS
# ============================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

# ============================================
# TRAIN PHASE 2
# ============================================

history_phase2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=PHASE2_EPOCHS,
    callbacks=[early_stop, lr_scheduler]
)

# ============================================
# SAVE MODEL
# ============================================

model.save("final_inception_model_299.h5")

print("Model saved successfully!")

# ============================================
# PLOT TRAINING GRAPH
# ============================================

train_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']

epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(8,6))

plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.title("InceptionV3 Training Performance")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("training_accuracy_plot.png")
plt.show()
