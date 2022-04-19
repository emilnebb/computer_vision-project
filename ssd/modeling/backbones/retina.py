from collections import OrderedDict
import torch
from typing import Tuple, List
from torch import nn
import torchvision
from torchsummary import summary


# https://piazza.com/class/kyipdksfp9q1dn?cid=302

class RetinaNet(torch.nn.Module):
    """
    Backbone = FPN
    FPN goes on top of ResNet
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

        nn.ConvTranspose2d
        #FPN
        self.fpn = torchvision.ops.FeaturePyramidNetwork([256]*5, 256)
        self.x_fpn = OrderedDict()

        #remove 2 last layers of resnet
        self.resnet_without_classifier = nn.Sequential(
            *list(
                torchvision.models.resnet18(pretrained=True, progress=True)
                .children()
            )[:-2]
        )


        #ONLY USE SELECTED LAYERS OF THE RESNET
        self.resnet.conv1.stride = (4, 4)
        print(self.resnet)
        #self.resnet.conv1.in_channels = image_channels
        #self.resnet.conv1.out_channels = 256

        self.resnet.conv1.kernel_size = (1, 1)
        #self.resnet.conv1.padding = (0,0)


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
        From paper:
        RetinaNet uses feature pyramid levels P3 to P7, where P3 to P5 are computed
        from the output of the corresponding ResNet residual stage (C3 through C5)
        using top-down and lateral connections just as in [19], P6 is obtained via
        a 3×3 stride-2 conv on C5, and P7 is computed by apply- ing ReLU followed
        by a 3×3 stride-2 conv on P6. This differs slightly from [19]: (1) we don’t
        use the high-resolution pyramid level P2 for com- putational reasons,
        (2) P6 is computed by strided convolution instead of downsampling,
        and (3) we include P7 to improve large object detection.
        These minor modifications improve speed while maintaining accuracy.
        """
        print(x.shape)
        self.resnet.conv1.in_channels = 64
        self.resnet.eval()
        #x = self.resnet(x)
        #print(x.shape)
        print("Conv 1:")

        self.resnet.conv1.in_channels = 3
        self.resnet.conv1.out_channels = 256
        #self.resnet.conv1.padding = (0, 0)
        self.resnet.conv1.kernel_size = (1, 1)
        self.resnet.conv1.stride = (2, 2)
        y = self.resnet.conv1(x)
        print(y.shape)

        print("Layer 1:")
        y = self.resnet.layer1(y) #keeps the same shape
        print(y.shape)
        print("Layer 1:") # keeps the same shape
        print(self.resnet.layer1.children())
        print(y.shape)
        print("Layer 2:")
        y = self.resnet.layer2(y)
        print(y.shape)
        print("Layer 3:")
        y = self.resnet.layer3(y)
        print(y.shape)
        print("Layer 3:")
        y = self.resnet.layer3[1](y)
        print(y.shape)

        """
        self.x_fpn['P_2'] = self.transform_stride_4(x) # torch.Size([1, 256, 32, 256])
        self.x_fpn['P_3'] = self.transform_stride_2(self.x_fpn['P_2']) # torch.Size([1, 256, 16, 128])
        self.x_fpn['P_4'] = self.transform_stride_2(self.x_fpn['P_3']) # torch.Size([1, 256, 8, 64])
        self.x_fpn['P_5'] = self.transform_stride_2(self.x_fpn['P_4']) # torch.Size([1, 256, 4, 32])
        self.x_fpn['P_6'] = self.transform_stride_2(self.x_fpn['P_5']) # torch.Size([1, 256, 2, 16])
        self.x_fpn['P_7'] = self.transform_stride_2(self.x_fpn['P_6']) # torch.Size([1, 256, 1, 8])
        """

        # x_fpn['P_2'] = self.transform_stride_4(x) # torch.Size([1, 256, 32, 256])
        x_fpn['P_3'] = self.transform_stride_8(x)  # torch.Size([1, 256, 16, 128])
        x_fpn['P_4'] = self.transform_stride_2(x_fpn['P_3'])  # torch.Size([1, 256, 8, 64])
        x_fpn['P_5'] = self.transform_stride_2(x_fpn['P_4'])  # torch.Size([1, 256, 4, 32])
        x_fpn['P_6'] = self.transform_stride_2(x_fpn['P_5'])  # torch.Size([1, 256, 2, 16])
        x_fpn['P_7'] = self.transform_stride_2(x_fpn['P_6'])  # torch.Size([1, 256, 1, 8])

        outputs = self.fpn(self.x_fpn).values()
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
