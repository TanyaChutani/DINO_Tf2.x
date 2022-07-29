import tensorflow as tf


class Dino(tf.keras.models.Model):
    def __init__(
        self,
        head,
        teacher_model,
        student_model,
        student_weights=None,
        teacher_weights=None,
    ):
        super(Dino, self).__init__()
        self.student_model = MultiCropWrapper(
            backbone=student_model, head=head, weights=student_weights
        )
        self.teacher_model = MultiCropWrapper(
            backbone=teacher_model, head=head, weights=teacher_weights
        )

    def compile(self, optimizer, dino_loss):
        super(Dino, self).compile()
        self.optimizer = optimizer
        self.dino_loss = dino_loss

    def train_step(self, data):
        global_image, local_image = data
        local_image = sum(local_image, ())
        global_image = sum(global_image, ())

        local_image = tf.stack(local_image)
        global_image = tf.stack(global_image)

        with tf.GradientTape() as tape:
            teacher_output = self.teacher_model(global_image)
            student_output = self.student_model(local_image)
            loss = tf.reduce_mean(self.dino_loss(student_output, teacher_output))
            student_gradients = tape.gradient(
                loss, self.student_model.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(student_gradients, self.student_model.trainable_variables)
            )
            return {"loss": loss}

    def test_step(self, data):
        global_image, local_image = data

        local_image = sum(local_image, ())
        global_image = sum(global_image, ())
        local_image = tf.stack(local_image)
        global_image = tf.stack(global_image)

        teacher_output = self.teacher_model(global_image, training=False)
        student_output = self.student_model(local_image, training=False)

        loss = tf.reduce_mean(self.dino_loss(student_output, teacher_output))

        return {"loss": loss}

    def call(self, image):
        output = self.teacher_model(image, training=False)
