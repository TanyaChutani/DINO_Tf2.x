import tensorflow as tf

from loss.dino_loss import DinoLoss


class Dino(tf.keras.models.Model):
    def __init__(
        self, teacher_model, student_model, student_weights=None, teacher_weights=None
    ):
        super(Dino, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.student_weights = student_weights
        self.teacher_weights = teacher_weights
        self.dino_loss = DinoLoss()

    def compile(self, optimizer):
        super(Dino, self).compile()
        self.optimizer = optimizer

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
