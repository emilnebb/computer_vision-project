from ast import Tuple
import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms
import math


class RetinaNet(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 anchors,
                 loss_objective,
                 num_classes: int,
                 anchor_prob_initialization: bool,
                 anchor_background_prob):
        super().__init__()
        """
            Implements the SSD network.
            Backbone outputs a list of features, which are gressed to SSD output with regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.regression_heads = []
        self.classification_heads = []
        self.anchor_prob_initialization = anchor_prob_initialization
        self.p = anchor_background_prob

        # Initialize output heads that are applied to each feature map from the backbone.
        for n_boxes, out_ch in zip(anchors.num_boxes_per_fmap, self.feature_extractor.out_channels):
            # in task 2.3 we will replace these heads with deeper convolutional nets
            self.regression_heads.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=out_ch,
                    out_channels=out_ch,  # is 4 the coordinates?!
                    kernel_size=3,
                    padding=1
                ),
                nn.Conv2d(
                    in_channels=out_ch,
                    out_channels=out_ch,  # is 4 the coordinates?!
                    kernel_size=3,
                    padding=1
                ),
                nn.Conv2d(
                    in_channels=out_ch,
                    out_channels=n_boxes * 4,  # is 4 the coordinates?!
                    kernel_size=3,
                    padding=1
                )
            ))
            self.classification_heads.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=out_ch, out_channels=out_ch,
                    kernel_size=3,
                    padding=1
                ),
                nn.Conv2d(
                    in_channels=out_ch, out_channels=out_ch,
                    kernel_size=3,
                    padding=1
                ),
                nn.Conv2d(
                    in_channels=out_ch, out_channels=n_boxes * self.num_classes,
                    kernel_size=3,
                    padding=1
                )
            ))

        self.regression_heads = nn.ModuleList(self.regression_heads)
        self.classification_heads = nn.ModuleList(self.classification_heads)
        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()

    def _init_weights(self):
        """
        initializes weights of the heads
        """
        #Improved weight initialization
        if self.anchor_prob_initialization:
            layers = [*self.regression_heads]
            for layer in layers:
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)

            layers = [*self.classification_heads]
            for layer in layers:
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)
                # Initialize biases of the last convolutional layers
                #print(f"Last layer = {layer[-1]}")
                nn.init.constant_(layer[-1].bias, 0)
                numbers_to_change = int(list(layer[-1].bias.shape)[0]/self.num_classes)
                #print(f"Numbers to change = {numbers_to_change}")
                nn.init.constant_(layer[-1].bias[:numbers_to_change], math.log(self.p*(self.num_classes-1)/(1-self.p)))
                #print(f"Bias shape = {layer[-1].bias.shape}")
                #print(f"Bias = {layer[-1].bias}")

        #Regular weight initialization
        else:
            layers = [*self.regression_heads, *self.classification_heads]
            for layer in layers:
                for param in layer[-1].parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)

    def regress_boxes(self, features: Tuple):
        locations = []
        confidences = []

        # iterate over all features (each returned by different FPN levels)
        # for each "resolution" we use individual regression- and classification heads

        for idx, x in enumerate(features):
            # print(f"feature shape={x.shape}")
            bbox_delta = self.regression_heads[idx](x).view(x.shape[0], 4,
                                                            -1)  # forward feature map through head and reshape it to : it's delta because its the offset fromt the default box. xmin_d, xmax_d, ymin_d, ymax_d
            # print(f"bbox_delta={bbox_delta.shape}")
            bbox_conf = self.classification_heads[idx](x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)

        bbox_delta = torch.cat(locations, 2).contiguous()  # N x 4 x num anchors
        confidences = torch.cat(confidences, 2).contiguous()  # N x num classes x num anchors
        return bbox_delta, confidences

    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)

        # as defined in the cfg.anchors.feature_sizes
        # features[0]: torch.Size([1, 256, 32, 256])
        # features[1]: torch.Size([1, 256, 16, 128])
        # features[2]: torch.Size([1, 256, 8, 64])
        # features[3]: torch.Size([1, 256, 4, 32])
        # features[4]: torch.Size([1, 256, 2, 16])
        # features[5]: torch.Size([1, 256, 1, 8])

        return self.regress_boxes(features)

    def forward_test(self,
                     img: torch.Tensor,
                     imshape=None,
                     nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions


def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
    """
        boxes_ltrb: shape [N, 4]
        confs: shape [N, num_classes]
    """
    assert 0 <= nms_iou_threshold <= 1
    assert max_output > 0
    assert 0 <= score_threshold <= 1
    scores, category = confs.max(dim=1)

    # 1. Remove low confidence boxes / background boxes
    mask = (scores > score_threshold).logical_and(category != 0)
    boxes_ltrb = boxes_ltrb[mask]
    scores = scores[mask]
    category = category[mask]

    # 2. Perform non-maximum-suppression
    keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

    # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
    keep_idx = keep_idx[:max_output]
    return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]