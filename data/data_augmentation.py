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

    def _crop_resize(self, image, mode="global"):
        scalee = self.global_crops_scale if mode == "global" else self.local_crops_scale
        final_size = (
            self.global_image_size if mode == "global" else self.local_image_size
        )
        height, width, channels = tf.shape(image)
        scaling_hw = tf.cast(tf.concat([height, width], axis=0), tf.float32)
        scale = tf.multiply(scalee, scaling_hw)
        scale = (
            tf.cast(scale[0].numpy(), tf.int32),
            tf.cast(scale[1].numpy(), tf.int32),
            channels,
        )
        image = tf.image.random_crop(value=image, size=scale)
        image = tf.image.resize(image, final_size, method="bicubic")
        return image

    def _apply_aug(self, image, mode="global"):
        image = self.flip_aug(image)
        image = self._crop_resize(image, mode)
        image = self._standardize_normalize(image)

        return image

    def __call__(self, image):
        crops = []
        crops.append(self._apply_aug(image))
        crops.append(self._apply_aug(image))
        for _ in range(self.local_crops_number):
            crops.append(self._apply_aug(image, mode="local"))
        return crops

