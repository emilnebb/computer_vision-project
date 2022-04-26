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


        # remove the 3 last layers of resnet
        self.resnet = torchvision.models.resnet18(pretrained=True, progress=True)

        # initialize fpn
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256, 512, 512, 256], 
            out_channels=256)

        # Focal Loss footnotes
        # P6 = 3×3 stride-2 conv on C5
        self.transform_c5_to_p6 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6
        self.transform_p6_to_p7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1 # not sure if this should be..
            ),
        )

        self.transform_64_to_256_channels = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=1
        )

        self.transform_128_to_256_channels = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=1
        )

        self.transform_512_to_256_channels = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=1
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

        # print(f"input shape={x.shape}")

        # manually apply the first four stages of ResNet
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # from here on we extract the next stages from resnet
        stage_1 = self.resnet.layer1(x)       # should be torch.Size([1, 256, 32, 256])
        stage_2 = self.resnet.layer2(stage_1) # should be torch.Size([1, 256, 16, 128])
        stage_3 = self.resnet.layer3(stage_2) # should be torch.Size([1, 256, 8, 64])
        stage_4 = self.resnet.layer4(stage_3) # should be torch.Size([1, 256, 4, 32])

        # we use convolution to further sample down to the desired dimensions
        downsample_1 = self.transform_c5_to_p6(stage_4)      # should be torch.Size([1, 256, 2, 16])
        downsample_2 = self.transform_p6_to_p7(downsample_1) # should be torch.Size([1, 256, 1, 8])

        # print(f"shape after init={x.shape}")

        x_fpn = OrderedDict()
        x_fpn['P_2'] = stage_1
        x_fpn['P_3'] = stage_2 
        x_fpn['P_4'] = stage_3 
        x_fpn['P_5'] = stage_4

        x_fpn['P_6'] = downsample_1
        x_fpn['P_7'] = downsample_2

        # # print(f"shapes of x_fpn={[(k, v.shape) for k, v in x_fpn.items()]}")

        out_features = self.fpn(x_fpn)
        print(f"shapes of fpn outputs={[v.shape for v in out_features]}")

        out_features_list = []

        for k, v in out_features.items():
            out_features_list.append(v)


        # # when not using fpn
        # out_features = [
        #     self.transform_64_to_256_channels(stage_1),
        #     self.transform_128_to_256_channels(stage_2),
        #     stage_3,
        #     self.transform_512_to_256_channels(stage_4),
        #     self.transform_512_to_256_channels(downsample_1),
        #     downsample_2
        # ]

        # for out_feature in out_features:
        #     print(out_feature.shape)


        # expected out DIMs:
        # IDX=0 Expected shape: (256, 32, 256)
        # IDX=1 Expected shape: (256, 16, 128)
        # IDX=2 Expected shape: (256, 8, 64)
        # IDX=3 Expected shape: (256, 4, 32)
        # IDX=4 Expected shape: (256, 2, 16)
        # IDX=5 Expected shape: (256, 1, 8)

        for idx, feature in enumerate(out_features_list):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features_list) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features_list)}"
        return tuple(out_features_list)

