import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


lists = ['ambulance']
for li in lists:
    path = "project/OIDv6/validation/"+li+"/labels"
    files = os.listdir(path)
print("done")


train_data=ImageDataGenerator(
    rescale=(1/255.),
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20,
    brightness_range=[0.8,1.2],
    horizontal_flip=True,
    )

test_data=ImageDataGenerator(
    rescale=(1/255.)
    )

validation_data=ImageDataGenerator(
    rescale=(1/255.)
    )

traindir = "project/OIDv6/train" 
testdir = "project/OIDv6/test"
valtdir = "project/OIDv6/validation"

train_generator=train_data.flow_from_directory(
    traindir,
    target_size =(224, 224),
    class_mode='categorical',
    batch_size=32
    )

test_generator=test_data.flow_from_directory(
    testdir,
    target_size =(224, 224),
    class_mode='categorical',
    batch_size=32
    )

validation_generator=test_data.flow_from_directory(
    valtdir,
    target_size =(224, 224),
    class_mode='categorical',
    batch_size=32
    )

class_weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),train_generator.classes)


target_labels = next(os.walk(traindir))[1]

target_labels.sort()

batch = next(train_generator)
batch_images = np.array(batch[0])
batch_labels = np.array(batch[1])

target_labels = np.asarray(target_labels)




IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()
len(base_model.layers)



image_batch, label_batch = next(iter(train_generator))

feature_batch = base_model(image_batch)


base_model.trainable = False



model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(5, activation='softmax')
])




base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


initial_epochs = 5
history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=validation_generator)


base_model.trainable = True

print("Total Layer on Based Model: ", len(base_model.layers))

fine_tune_at = 123

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False



lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    base_learning_rate,
    decay_steps=50,
    decay_rate=0.9)



model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              metrics=['accuracy'])

model.summary()


checkpoint_path = "new_checkpoint/cp_rev_1.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



fine_tune_epochs = 25
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_generator,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_generator,
                         callbacks=[cp_callback])



acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']


model.save_weights('Checkpoint_ver2/Ambulance_checkpoint')
loss, accuracy = model.evaluate(test_generator)
print('Test accuracy :', accuracy)
loss, accuracy = model.evaluate(validation_generator)
print('Validation accuracy :', accuracy)
model.save('THE_model',save_format='h5')