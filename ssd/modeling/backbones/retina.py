from collections import OrderedDict
import torch
from typing import Tuple, List
from torch import nn
import torchvision
from torchsummary import summary


# https://piazza.com/class/kyipdksfp9q1dn?cid=302

class RetinaNet(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    """

    def __init__(self,
                 output_channels: List[int],
                 image_channels: int,
                 output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        # this transform layer brings the input to the right shape for the different levels of the fpn
        self.transform_stride_8 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=256,
            kernel_size=1,
            stride=8,
            padding=0
        )

        self.transform_stride_2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=2,
            padding=0
        )

        self.transform_out_channels_3 = nn.Conv2d(
            in_channels=256,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.fpn = torchvision.ops.FeaturePyramidNetwork([256] * 5, 256)
        self.resnet_without_classifier = nn.Sequential(
            *list(
                torchvision.models.resnet18(pretrained=True, progress=True)
                .children()
            )[:-2]
        )

        self.transpose_convolution = nn.ConvTranspose2d(
            in_channels=512, 
            out_channels=256, 
            kernel_size=1, 
            #stride=1,
            padding=1,
            #dilation=???
        )

    def forward(self, x):
        """

        """

        x_fpn = OrderedDict()

        # x_fpn['P_2'] = self.transform_stride_4(x) # torch.Size([1, 256, 32, 256])
        x_fpn['P_3'] = self.transform_stride_8(x)  # torch.Size([1, 256, 16, 128])
        x_fpn['P_4'] = self.transform_stride_2(x_fpn['P_3'])  # torch.Size([1, 256, 8, 64])
        x_fpn['P_5'] = self.transform_stride_2(x_fpn['P_4'])  # torch.Size([1, 256, 4, 32])
        x_fpn['P_6'] = self.transform_stride_2(x_fpn['P_5'])  # torch.Size([1, 256, 2, 16])
        x_fpn['P_7'] = self.transform_stride_2(x_fpn['P_6'])  # torch.Size([1, 256, 1, 8])

        # outputs = self.fpn(x_fpn).values()

        # print(outputs_fpn.values().next().shape)
        # outputs = [self.transform_out_channels_3(output) for output in outputs]
        print(f"input={x.shape}")

        # # set model in evaluation mode
        # self.resnet.eval()

        # outputs = [self.resnet(output) for output in outputs]

        outputs = self.resnet_without_classifier(x)
        print(f"output resnet={outputs.shape}")

        # use "transposed convolution" to increase the output dimension of ResNet to fit our req. dims
        outputs = self.transpose_convolution(outputs)
        print(f"output transpose={outputs.shape}")

        summary(self.resnet_without_classifier,input_size=(3, 128, 1024))

        exit(0)

        # expected out DIMs:
        # IDX=0 Expected shape: (256, 32, 256)
        # IDX=1 Expected shape: (256, 16, 128)
        # IDX=2 Expected shape: (256, 8, 64)
        # IDX=3 Expected shape: (256, 4, 32)
        # IDX=4 Expected shape: (256, 2, 16)
        # IDX=5 Expected shape: (256, 1, 8)

        for idx, feature in enumerate(outputs):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(outputs) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(outputs)}"
        return tuple(outputs)
