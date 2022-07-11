import tensorflow as tf


class MultiCropWrapper(tf.keras.layers.Layer):
    def __init__(self, head, backbone):
        super(MultiCropWrapper, self).__init__()
        self.head = head
        self.backbone = backbone

    @staticmethod
    def unique_consecutive(x):
        neq = tf.not_equal(x[1:], x[:-1])
        cum_sum = tf.cumsum(tf.dtypes.cast(neq, tf.int32))
        return tf.concat([[0], cum_sum], axis=0)

    def call(self, x):
        if not isinstance(x, list):
            x = [x]
        unq = tf.constant([inp.shape[0] for inp in x], dtype=tf.int32)
        unique_idx = self.unique_consecutive(unq)
        _, _, idx_crops = tf.unique_with_counts(unique_idx)
        idx_crops = tf.cumsum(idx_crops) + 1
        start_idx, output = tf.constant(0), tf.reshape(
            tf.convert_to_tensor(()), (0, 768)
        )
        for end_idx in idx_crops:
            _out = self.backbone(
                x[tf.get_static_value(start_idx) : tf.get_static_value(end_idx)]
            )
            if isinstance(_out, tuple):
                _out = _out[0]
            output = tf.concat([output, _out], axis=0)
            start_idx = end_idx
        return self.head(output)
