import tensorflow as tf
from loss.utils import TeacherTemp


class DinoLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        nepochs=100,
        out_dim=65536,
        ncrops=2,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=30,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super(DinoLoss, self).__init__()
        self.ncrops = ncrops
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        self.teacher_temp_schedule = tf.concat(
            (
                tf.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                tf.ones((nepochs - warmup_teacher_temp_epochs)) * teacher_temp,
            ),
            axis=0,
        )

    def update_center(self, teacher_output):
        batch_center = tf.math.reduce_sum(teacher_output, axis=0)
        batch_center = batch_center / tf.cast(len(teacher_output), tf.float32)
        self.center = tf.stop_gradient(
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

    def call(self, student_output, teacher_output):
        teacher_output = tf.cast(teacher_output, tf.float32)
        student_output = tf.cast(student_output, tf.float32)

        student_out = student_output / self.student_temp
        student_out = tf.split(student_out, num_or_size_splits=self.ncrops)

        self.center = tf.zeros_like(teacher_output, dtype=tf.float32)
        teacher_out = tf.stop_gradient(
            tf.nn.softmax(
                (teacher_output - self.center) / TeacherTemp(0.04).temp, axis=-1
            )
        )
        teacher_out = tf.split(
            tf.tile(teacher_out, tf.constant([2, 1], tf.int32)), num_or_size_splits=1
        )

        total_loss = 0
        n_loss_terms = 0
        for idx, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                q = tf.stop_gradient(q)
                if v == idx:
                    continue
                loss = tf.reduce_sum(
                    -q * tf.nn.log_softmax(student_out[v], axis=-1), axis=-1
                )
                total_loss += tf.math.reduce_mean(loss)
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
