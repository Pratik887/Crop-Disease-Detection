import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


img_size = (128, 128)
batch_size = 32


train_dir = "dataset/train"


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_data = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)


print("Class Indices:", train_data.class_indices)


model = models.Sequential([

 
    layers.Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(128,128,3)
    ),
    layers.MaxPooling2D(2,2),

 
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),


    layers.Flatten(),

 
    layers.Dense(128, activation="relu"),

  
    layers.Dense(3, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


model.save("model.h5")

print("Model saved successfully!")
