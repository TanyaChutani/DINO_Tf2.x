import argparse
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize

from model.dino import Dino
from model.dino_head import DinoHead
from model.utils import MultiCropWrapper, load_base


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-image_size", "--image_size", type=int, metavar="", default=224
    )
    parser.add_argument(
        "-wp", "--weights_path", type=str, metavar="", default="techer_weights"
    )
    parser.add_argument("-ct", "--crop_teacher", type=int, metavar="", default=224)
    parser.add_argument("-cs", "--crop_student", type=int, metavar="", default=96)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    head = DinoHead()

    student = load_base(args.crop_student)
    teacher = load_base(args.crop_teacher)

    student = MultiCropWrapper(backbone=student, head=head)
    teacher = MultiCropWrapper(backbone=teacher, head=head)

    model = Dino(teacher, student, teacher_weights=args.weights_path)

    vit = model.teacher_model.get_layer("vit-b16")

    url = "https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg"
    image = utils.read(url, args.image_size)

    attention_map = visualize.attention_map(model=vit, image=image)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.set_title("Original")
    ax2.set_title("Attention Map")
    _ = ax1.imshow(image)
    _ = ax2.imshow(attention_map)
