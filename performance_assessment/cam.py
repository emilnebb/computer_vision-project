import cv2
import os
import tops
import click
import numpy as np
from tops.config import instantiate
from tops.config import LazyCall as L
from tops.checkpointer import load_checkpoint
from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm
from ssd.data.transforms import ToTensor

from performance_assessment.save_comparison_images import get_config, get_trained_model, get_dataloader, convert_boxes_coords_to_pixel_coords, convert_image_to_hwc_byte, create_filepath, create_comparison_image, create_and_save_comparison_images

def predict():
    #pred_image = tops.to_cuda(batch["image"])
    #transformed_image = img_transform({"image": pred_image})["image"]
    #boxes, categories, scores = model(transformed_image, score_threshold=score_threshold)[0]
    return 0

def get_save_folder_name(cfg):
    return os.path.join(
        "performance_assessment",
        cfg.run_name,
        "cam"
    )

@click.command()
@click.argument("config_path")
@click.option("--train", default=False, is_flag=True, help="Use the train dataset instead of val")
@click.option("-n", "--num_images", default=500, type=int, help="The max number of images to save")
@click.option("-c", "--conf_threshold", default=0.3, type=float, help="The confidence threshold for predictions")
def main(config_path, train, num_images, conf_threshold):
    cfg = get_config(config_path)
    model = get_trained_model(cfg)

    if train:
        dataset_to_visualize = "train"
    else:
        dataset_to_visualize = "val"

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    img, _ = next(iter(iter(dataloader)))
    img.show()

    #create_and_save_comparison_images(dataloader, model, cfg, save_folder, conf_threshold, num_images)

    #predict()

if __name__ == '__main__':
    main()