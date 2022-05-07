import PIL.Image
import cv2
import os
import tops
import click
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tops.config import instantiate
from tops.config import LazyCall as L
from tops.checkpointer import load_checkpoint
#from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm
from ssd.data.transforms import ToTensor
import torchvision

from performance_assessment.save_comparison_images import get_config, get_trained_model, get_dataloader, convert_boxes_coords_to_pixel_coords, convert_image_to_hwc_byte, create_filepath, create_comparison_image, create_and_save_comparison_images

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import requests
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image, preprocess_image


#Piazza thread for grad grad_cam:
#https://piazza.com/class/kyipdksfp9q1dn?cid=420

#Turorial:
#https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.ipynb

#Wrapper for model:
#https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Semantic%20Segmentation.ipynb

class RetinaNetModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        dict_from_outputs = OrderedDict()
        b, l, s = self.model(x)[0]
        dict_from_outputs["boxes"] = Variable(b, requires_grad=True)
        dict_from_outputs["labels"] = l
        dict_from_outputs["scores"] = Variable(s, requires_grad=True)
        return [dict_from_outputs]


def fasterrcnn_reshape_transform(x):
    target_size = x[3].shape[-2 : ]
    #print(f"Target size {target_size}")
    activations = []
    for value in x:
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

def get_save_folder_name(cfg):
    return os.path.join(
        "performance_assessment",
        cfg.run_name,
        "grad_cam"
    )

@click.command()
@click.argument("config_path")
@click.option("--train", default=False, is_flag=True, help="Use the train dataset instead of val")
@click.option("-n", "--num_images", default=500, type=int, help="The max number of images to save")
@click.option("-c", "--conf_threshold", default=0.3, type=float, help="The confidence threshold for predictions")
def main(config_path, train, num_images, conf_threshold):
    cfg = get_config(config_path)

    #Collecting our model, comes in eval mode
    model = get_trained_model(cfg)

    torch.set_grad_enabled(True)

    if train:
        dataset_to_visualize = "train"
    else:
        dataset_to_visualize = "val"

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    img_tensor = next(iter(dataloader))['image'] #tensor
    #print(f"Image tensor shape {img_tensor.shape}")

    img_numpy = img_tensor[0].numpy().transpose(1, 2, 0)
    img_tensor = Variable(img_tensor, requires_grad=True)
    #plt.imshow(img_numpy)
    #plt.show()
    image_float_np = np.float32(img_numpy) / 255

    model1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    test_target_layers = [model1.backbone]
    #print(model1)
    #print(f"Test_target_layers = {test_target_layers}")

    boxes, labels, scores = model(img_tensor)[0]
    boxes = Variable(boxes, requires_grad=True)


    wrapped_model = RetinaNetModelOutputWrapper(model)
    wrapped_model.model.eval()

    target_layers = [model.feature_extractor]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(wrapped_model,
                   target_layers,
                   use_cuda=torch.cuda.is_available(),
                   reshape_transform=fasterrcnn_reshape_transform)

    grayscale_cam = cam(img_tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    #image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    Image.fromarray(cam_image)





if __name__ == '__main__':
    main()