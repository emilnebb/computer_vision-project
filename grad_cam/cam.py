import PIL.Image
import cv2
import os
import tops
import click
import numpy as np
from collections import OrderedDict


from performance_assessment.save_comparison_images import get_config, get_trained_model, get_dataloader
import torchvision
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image

#Command line to run this script:
# python -m grad_cam.cam configs/tdt4265.py

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
        dict_from_outputs["boxes"] = b
        dict_from_outputs["labels"] = l
        dict_from_outputs["scores"] = s
        return [dict_from_outputs]

coco_names = ['__background__', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person']

def fasterrcnn_reshape_transform(x):
    #Pick first layer in feature map to get more fine grained visualization
    target_size = x[0].shape[-2 : ]
    activations = []
    for value in x:
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

def predict(input_tensor, model):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        boxes.append(pred_bboxes[index].astype(np.int32))
        classes.append(pred_classes[index])
        labels.append(pred_labels[index])
        indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

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

    if train:
        dataset_to_visualize = "train"
    else:
        dataset_to_visualize = "val"

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    dataloader = iter(dataloader)

    #Iterate into the datset and pick a picture
    for i in range(0,150):
        img_tensor = next(dataloader)['image'] #tensor

    img_numpy = img_tensor[0].numpy().transpose(1, 2, 0)
    img_tensor = tops.to_cuda(Variable(img_tensor, requires_grad=True))

    wrapped_model = RetinaNetModelOutputWrapper(model)
    boxes, classes, labels, indices = predict(img_tensor, wrapped_model)

    target_layers = [model.feature_extractor]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    cam = EigenCAM(wrapped_model,
                   target_layers,
                   use_cuda=torch.cuda.is_available(),
                   reshape_transform=fasterrcnn_reshape_transform)

    #Turn off gradient calculations
    cam.uses_gradients = False

    grayscale_cam = cam(img_tensor, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(img_numpy, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    #print(f"Image type: {type(image_with_bounding_boxes)}")
    img = Image.fromarray(image_with_bounding_boxes)
    img.save("grad_cam/grad_cam_pictures/grad150_0.png")
    img.show()





if __name__ == '__main__':
    main()