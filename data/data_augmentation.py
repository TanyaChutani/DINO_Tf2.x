import tensorflow as tf


class DataAugmentationDino:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_image_size=[224, 224],
        local_image_size=[96, 96],
        mean=[0.485, 0.456, 0.406],
        std_dev=[0.229, 0.224, 0.225],
    ):
        self.mean = mean
        self.std_dev = std_dev
        self.local_image_size = local_image_size
        self.global_image_size = global_image_size
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_scale = global_crops_scale

        self.flip_aug = tf.keras.Sequential(
            [tf.keras.layers.RandomFlip(mode="horizontal")]
        )

    def _standardize_normalize(self, image):
        image = image / 255.0
        image -= self.mean
        image /= self.std_dev
        image = tf.cast(image, tf.float32)
        return image

    def _color_jitter(image):
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.0, upper=0.4)
        image = tf.image.random_saturation(image, lower=0.0, upper=0.2)
        image = tf.image.random_hue(image, max_delta=0.1)
        return image

