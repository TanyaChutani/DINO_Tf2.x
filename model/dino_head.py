import tensorflow as tf
import tensorflow_addons as tfa


class DinoHead(tf.keras.models.Model):
    def __init__(
        self,
        nlayers=3,
        in_dim=768,
        out_dim=65536,
        use_bn=False,
        hidden_dim=2048,
        bottleneck_dim=256,
        norm_last_layer=True,
    ):
        super(DinoHead, self).__init__()
        self.in_dim = in_dim
        self.use_bn = use_bn
        self.out_dim = out_dim
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.norm_last_layer = norm_last_layer
        self.last_layer = tf.keras.layers.Dense(self.out_dim)

        self.mlp_block = self.mlp()

    def mlp(self):
        layer = []
        layer.append(tf.keras.layers.Dense(self.hidden_dim, input_shape=(self.in_dim,)))
        if self.use_bn:
            layer.append(tf.keras.layers.BatchNormalization())
        layer.append(tfa.layers.GELU())
