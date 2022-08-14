import tensorflow as tf
from vit_keras import vit, utils


class MultiCropWrapper(tf.keras.models.Model):
    def __init__(self, backbone, head, weights=None):
        super(MultiCropWrapper, self).__init__()
        self.head = head
        self.backbone = backbone
        if weights:
            try:
                print("Restoring model weights from: ", weights)
                self.load_weights(weights)
            except Exception:
                raise ValueError

    @staticmethod
    def unique_consecutive(x):
        neq = tf.math.not_equal(x, x)
        neq = tf.cast(neq, tf.int32)
        if neq.shape[0] > 1:
            neq = tf.math.cumsum(tf.cast(neq, tf.int32), axis=0)
        neq = tf.concat([[0], neq], axis=0)
        _, _, count = tf.unique_with_counts(neq)
        return count

    def call(self, x):
        if not isinstance(x, list):
            x = [x]
        unq = tf.constant([inp.shape[0] for inp in x], dtype=tf.int32)
        count = self.unique_consecutive(unq)
        start_idx, output = tf.constant(0), tf.zeros((0, 768), dtype=tf.float32)
        for end_idx in count:
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(output, tf.TensorShape([None, None]))]
            )
            _out = self.backbone(
                x[tf.get_static_value(start_idx) : tf.get_static_value(end_idx)]
            )
            if isinstance(_out, tuple):
                _out = _out[0]
            output = tf.concat([output, _out], axis=0)
            start_idx = end_idx
        return self.head(output)


def load_base(image_size, include_pretrained=True):
    model = vit.vit_b16(
        image_size=image_size,
        pretrained=include_pretrained,
        pretrained_top=False,
        include_top=False,
    )
    return model
