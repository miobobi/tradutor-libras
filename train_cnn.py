import tensorflow as tf
import os
os.makedirs('models', exist_ok=True)

DATA_DIR = 'data/images'
BATCH = 32
IMG_SIZE = (224,224)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='training', seed=42, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='validation', seed=42, image_size=IMG_SIZE, batch_size=BATCH)

base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(*IMG_SIZE,3), weights='imagenet')
base.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint('models/cnn_best.h5', save_best_only=True)]
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

# fine-tune
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

model.save('models/cnn_final.h5')
print('Saved CNN model to models/cnn_final.h5')
