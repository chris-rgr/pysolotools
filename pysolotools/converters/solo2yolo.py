import argparse
import multiprocessing
import shutil
import sys
import numpy as np
from pathlib import Path

from pysolotools.consumers import Solo
from pysolotools.core import BoundingBox2DAnnotation, RGBCameraCapture
from pysolotools.core.models import Frame


class Solo2YoloConverter:
    def __init__(self, solo: Solo):
        self._solo = solo
        self._pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    @staticmethod
    def _filter_annotation(ann_type, annotations):
        filtered_ann = list(
            filter(
                lambda k: isinstance(
                    k,
                    ann_type,
                ),
                annotations,
            )
        )
        if filtered_ann:
            return filtered_ann[0]
        return filtered_ann

    @staticmethod
    def _process_rgb_image(image_id, rgb_capture, output, data_root, sequence_num):
        image_file = data_root / f"sequence.{sequence_num}/{rgb_capture.filename}"
        image_to_file = f"camera_{image_id}.png"
        image_to = output / image_to_file
        shutil.copy(str(image_file), str(image_to))

    @staticmethod
    def _to_yolo_bbox(img_width, img_height, center_x, center_y, box_width, box_height):
        x = (center_x + box_width * 0.5) / img_width
        y = (center_y + box_height * 0.5) / img_height
        w = box_width / img_width
        h = box_height / img_height

        return x, y, w, h

    @staticmethod
    def _process_annotations(image_id, rgb_capture, output):
        width, height = rgb_capture.dimension
        filename = f"camera_{image_id}.txt"
        file_to = output / filename

        bbox_ann = Solo2YoloConverter._filter_annotation(
            ann_type=BoundingBox2DAnnotation, annotations=rgb_capture.annotations
        ).values

        with open(str(file_to), "w") as f:
            for bbox in bbox_ann:
                x, y, w, h = Solo2YoloConverter._to_yolo_bbox(
                    width,
                    height,
                    bbox.origin[0],
                    bbox.origin[1],
                    bbox.dimension[0],
                    bbox.dimension[1],
                )
                f.write(f"{bbox.labelId} {x} {y} {w} {h}\n")


    @staticmethod
    def _generate_split_assignment(train_split, val_split, test_split, n_frames):
        train_split = int(n_frames * train_split)
        val_split = int(n_frames * val_split)
        test_split = int(n_frames * test_split)

        assignment_array = np.zeros(n_frames)
        assignment_array[:train_split] = 0
        assignment_array[train_split : train_split + val_split] = 1
        assignment_array[train_split + val_split :] = 2
        np.random.shuffle(assignment_array)

        return assignment_array

    @staticmethod
    def _process_instances(frame: Frame, idx, images_output, labels_output, data_root):
        image_id = idx
        sequence_num = frame.sequence

        # Currently support only single camera
        rgb_capture = list(
            filter(lambda cap: isinstance(cap, RGBCameraCapture), frame.captures)
        )[0]

        Solo2YoloConverter._process_rgb_image(
            image_id, rgb_capture, images_output, data_root, sequence_num
        )
        Solo2YoloConverter._process_annotations(image_id, rgb_capture, labels_output)

    def process_bbox_anotation_definition(self, output_path: str):
        config_output = output_path / "config.yaml"
        config_output.touch()

        bounding_box_definition = None
        for definition in self._solo.annotation_definitions.annotationDefinitions:
            if definition.id == "bounding box":
                bounding_box_definition = definition
                break

        classes_count = len(bounding_box_definition.spec)
        classes_names = [label_spec.label_name for label_spec in bounding_box_definition.spec]

        with open(str(config_output), "w") as f:
            f.write("train: ./images/train\n")
            f.write("val: ./images/val\n")
            f.write("test: ./images/test\n")
            f.write("nc: " + str(classes_count) + "\n")
            f.write("names: " + str(classes_names) + "\n")

    

    def convert(self, output_path: str, train_split, val_split, test_split):
        base_path = Path(output_path)

        images_output = base_path / "images"
        labels_output = base_path / "labels"
        split_suffixes = ["train", "val", "test"]

        # Ensuring directories exist, also for the splits
        images_output.mkdir(parents=True, exist_ok=True)
        labels_output.mkdir(parents=True, exist_ok=True)
        for split in split_suffixes:
            (images_output / split).mkdir(parents=True, exist_ok=True)
            (labels_output / split).mkdir(parents=True, exist_ok=True)


        data_path = Path(self._solo.data_path)

        self.process_bbox_anotation_definition(output_path)

        for index, frame in enumerate(self._solo.frames()):
            image_output_path = images_output / split_suffixes[index]
            label_output_path = labels_output / split_suffixes[index]
            self._pool.apply_async(
                self._process_instances,
                args=(frame, index, image_output_path, label_output_path, data_path),
            )

        self._pool.close()
        self._pool.join()


def cli():
    parser = argparse.ArgumentParser(
        prog="solo2yolo",
        description=("Converts SOLO datasets into YOLO datasets",),
        epilog="\n",
    )

    parser.add_argument("solo_path")
    parser.add_argument("yolo_path")
    parser.add_argument("training_split", type=float, default=0.8)
    parser.add_argument("validation_split", type=float, default=0.1)
    parser.add_argument("test_split", type=float, default=0.1)

    args = parser.parse_args(sys.argv[1:])

    solo = Solo(args.solo_path)

    training_split = args.training_split
    validation_split = args.validation_split
    test_split = args.test_split

    converter = Solo2YoloConverter(solo)

    converter.convert(args.yolo_path, training_split, validation_split, test_split)


if __name__ == "__main__":
    cli()
