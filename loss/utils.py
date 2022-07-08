import tensorflow as tf


class TeacherTemp(tf.keras.callbacks.Callback):
    def __init__(
        self,
        temp,
        nepochs=100,
        teacher_temp=0.04,
        warmup_teacher_temp=0.04,
        warmup_teacher_temp_epochs=30,
    ):
        super(TeacherTemp, self).__init__()
        self.temp = temp
        self.teacher_temp_schedule = tf.concat(
            (
                tf.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                tf.ones((nepochs - warmup_teacher_temp_epochs)) * teacher_temp,
            ),
            axis=0,
        )

    @tf.function
    def on_epoch_begin(self, epoch, logs={}):
        self.temp = tf.Variable(self.temp, trainable=True, dtype=tf.float32)
        tf.keras.backend.set_value(self.temp, self.teacher_temp_schedule[epoch])
        logs["temp"] = tf.keras.backend.get_value(self.temp)
