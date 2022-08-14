import argparse
import tensorflow as tf

from model.dino import Dino
from model.dino_head import DinoHead
from data.data_generator import DataGenerator
from model.utils import MultiCropWrapper, load_base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epoch", "--epochs", type=int, metavar="", default=100)
    parser.add_argument("-b", "--batch_size", type=int, metavar="", default=2)
    parser.add_argument("-ct", "--crop_teacher", type=int, metavar="", default=224)
    parser.add_argument("-cs", "--crop_student", type=int, metavar="", default=96)
    parser.add_argument(
        "-d_train",
        "--dataset_train",
        type=str,
        metavar="",
        default="VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
    )
    parser.add_argument(
        "-d_test",
        "--dataset_test",
        type=str,
        metavar="",
        default="VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
    )
    parser.add_argument(
        "-s_weights",
        "--student_weights_path",
        type=str,
        metavar="",
        default="student_weights",
    )
    parser.add_argument(
        "-t_weights",
        "--teacher_weights_path",
        type=str,
        metavar="",
        default="teacher_weights",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    head = DinoHead()

    student = load_base(args.crop_student)
    teacher = load_base(args.crop_teacher)

    student = MultiCropWrapper(backbone=student, head=head)
    teacher = MultiCropWrapper(backbone=teacher, head=head)

    model = Dino(teacher, student)

    train_dataset = DataGenerator(
        mode="train",
        dataset_path=args.dataset_train,
        batch_size=args.batch_size,
        local_image_size=args.crop_student,
        global_image_size=args.crop_teacher,
    )

    val_dataset = DataGenerator(
        mode="val",
        dataset_path=args.dataset_test,
        batch_size=args.batch_size,
        local_image_size=args.crop_student,
        global_image_size=args.crop_teacher,
    )

    learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[args.epochs / 2], values=[0.0001, 0.00001]
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.teacher_weights_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    model.build(input_shape=(1, args.crop_teacher, args.crop_teacher, 3))
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.student_model.save_weights(args.student_weights_path)
    model.teacher_model.save_weights(args.teacher_weights_path)


if __name__ == "__main__":
    main()
