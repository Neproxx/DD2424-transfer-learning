import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from typing import Union

# from torchvision.models.convnext import LayerNorm2d
# from functools import partial


class ConvNeXt(nn.Module):
    """
    This is a wrapper around the ConvNeXt model that allows for multi-task learning
    by adding an arbitrary number of output layers to the end of the network.

    Example:
    ConvNeXt([10, 20, 25]) will create a ConvNeXt model with three output layers
    that have 10, 20, and 25 output neurons, respectively.
    """

    def __init__(self, output_sizes: Union[list, int], device):
        """
        Args:
            output_sizes (list or int): An int or list of output sizes for each output layer (one per task).
        """
        super(ConvNeXt, self).__init__()

        self.device = device

        if isinstance(output_sizes, int):
            output_sizes = [output_sizes]

        assert len(output_sizes) > 0, "At least one output size must be specified"

        raw_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.features = raw_model.features
        self.avgpool = raw_model.avgpool

        # ConvNeXt has a 'classifier' module at the end that looks like this:
        # Sequential(
        #    (0): LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True)
        #    (1): Flatten(start_dim=1, end_dim=-1)
        #    (2): Linear(in_features=1024, out_features=1000, bias=True)
        #    )
        # The linear layer is what we need to replace with new output layers.
        # Therefore, we have to re-build the classifier module but replace
        # the linear layer with a linear layer for each task.
        # See also: https://github.com/pytorch/vision/blob/62c22317760f5aa0ff181a6ad7b3f801fa8639b6/torchvision/models/convnext.py#L98

        # NOTE: The LayerNorm2d layer is throws an error which may be due to version
        # difference. Therefore, we just remove it for now.
        # self.norm_layer = partial(
        #    LayerNorm2d, eps=1e-6, dtype=torch.float32, device=self.device()
        # )
        self.flatten = nn.Flatten(1)
        self.output_layers = nn.ModuleList(
            [nn.Linear(1024, output_size) for output_size in output_sizes]
        )

        self.to(self.device)

    def forward(self, x):
        # Operations that we have to repeat, because we replaced the classifier
        x = self.features(x)
        x = self.avgpool(x)
        # x = self.norm_layer(x)
        x = self.flatten(x)

        if len(self.output_layers) == 1:
            return self.output_layers[0](x)

        outputs = []
        for layer in self.output_layers:
            outputs.append(layer(x))
        return outputs

    def replace_fc(self, num_classes: int):
        """
        Replace the final fully connected layer with a new one. This is useful
        if you want to fine-tune the model on a specific task.

        Args:
            num_classes (int): Number of output neurons.
        """
        linear = nn.Linear(1024, num_classes, device=self.device)
        self.output_layers = nn.ModuleList([linear])
