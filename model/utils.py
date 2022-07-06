class MultiCropWrapper(tf.keras.layers.Layer):
    def __init__(self, head, backbone):
        super(MultiCropWrapper, self).__init__()
        self.head = head
        self.backbone = backbone

    @staticmethod
    def unique_consecutive(x):
        neq = tf.not_equal(x[1:], x[:-1])
        cum_sum = tf.cumsum(tf.dtypes.cast(neq, tf.int64))
        return tf.concat([[0], cum_sum], axis=0)
