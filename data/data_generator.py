import tensorflow as tf
import os

from data.data_augmentation import DataAugmentationDino


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        mode,
        batch_size,
        dataset_path,
        local_image_size,
        global_image_size,
        shuffle=True,
    ):
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.dataset = os.listdir(dataset_path)
        self.local_image_size = local_image_size
        self.global_image_size = global_image_size
        self.on_epoch_end()

    def _load_image(self, path, data_augmentation):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = data_augmentation(image) if self.mode == "train" else image
        return image

    def on_epoch_end(self):
        self.index = tf.range(len(self.dataset))
        if self.shuffle == True:
            tf.random.shuffle(self.index)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        datset_keys = [self.dataset[k] for k in indexes]
        (global_images, local_images) = self.__data_generation(datset_keys)
        return global_images, local_images

    def __data_generation(self, index):
        batch_global, batch_local = [], []
        dino = DataAugmentationDino((0.4, 1.0), (0.05, 0.4), 8)
        for idx, i in enumerate(index):
            images = self._load_image(os.path.join(self.dataset_path, i), dino)
            global_images = images[:2]
            # unable to stack varied size input in the dataset
            local_images = images[2:]
            batch_local.append(local_images)
            batch_global.append(global_images)
        return batch_global, batch_local
