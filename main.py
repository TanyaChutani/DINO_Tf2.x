import tensorflow as tf



model = Dino(teacher, student)

train_dataset = DataGenerator(
    mode="train",
    dataset_path="VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
    batch_size=2,
    local_image_size=96,
    global_image_size=224,
)

val_dataset = DataGenerator(
    mode="val",
    dataset_path="VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
    batch_size=2,
    local_image_size=96,
    global_image_size=224,
)
epochs = 10
weights_path = "dino"
learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[epochs / 2], values=[0.0001, 0.00001]
)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        weights_path,
        monitor="loss",
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
    )
]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
model.fit(train_dataset, validation_data = val_dataset, epochs = epochs, callbacks=callbacks)
