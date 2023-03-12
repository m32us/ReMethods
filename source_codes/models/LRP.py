from copy import deepcopy

import torch
from torch import nn

from models.layers import RelevancePropagationAdaptiveAvgPool2d, RelevancePropagationReLU, RelevancePropagationAvgPool2d, RelevancePropagationConv2d, RelevancePropagationDropout, RelevancePropagationFlatten, RelevancePropagationIdentity, RelevancePropagationLinear, RelevancePropagationMaxPool2d


def layers_lookup() -> dict:
    """Lookup table to map network layer to associated LRP operation.
    Returns:
        Dictionary holding class mappings.
    """
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
    }
    return lookup_table


class LRPModel(nn.Module):
    """Class wraps PyTorch model to perform layer-wise relevance propagation."""

    def __init__(self, model: torch.nn.Module, top_k: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.top_k = top_k

        self.model.eval()  # self.model.train() activates dropout / batch normalization etc.!

        # Parse network
        self.feature_layers, self.classifier_layers = self._get_layer_operations()

        # Create LRP network
        self.flrp_layers, self.clrp_layers = self._create_lrp_model()

    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """Method builds the model for layer-wise relevance propagation.
        Returns:
            LRP-model as module list.
        """
        # Clone layers from original model. This is necessary as we might modify the weights.
        flayers = deepcopy(self.feature_layers)
        clayers = deepcopy(self.classifier_layers)

        lookup_table = layers_lookup()

        # Run backwards through layers
        for i, layer in enumerate(clayers[::-1]):
            try:
                clayers[i] = lookup_table[layer.__class__](
                    layer=layer, top_k=self.top_k)
            except KeyError:
                message = (
                    f"Layer-wise relevance propagation not implemented for "
                    f"{layer.__class__.__name__} layer."
                )
                raise NotImplementedError(message)

        # Run backwards through layers
        for i, layer in enumerate(flayers):
            try:
                flayers[i] = lookup_table[layer.__class__](
                    layer=layer, top_k=self.top_k)
            except KeyError:
                message = (
                    f"Layer-wise relevance propagation not implemented for "
                    f"{layer.__class__.__name__} layer."
                )
                raise NotImplementedError(message)

        return flayers, clayers

    def _get_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.
        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.
        Returns:
            Layers of original model stored in module list.
        """
        feature_layers = torch.nn.ModuleList()
        classifier_layers = torch.nn.ModuleList()

        for layer in self.model.features:
            feature_layers.append(layer)

        for layer in self.model.classifier:
            classifier_layers.append(layer)

        return feature_layers, classifier_layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.
        Args:
            x: Input tensor representing an image / images (N, C, H, W).
        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).
        """
        activations = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.feature_layers:
                x = layer.forward(x)
                activations.append(x)
                print(x.shape)

            x = x.view(x.size(0), -1)
            activations.append(x)
            print(x.shape)

            for layer in self.classifier_layers:
                x = layer.forward(x)
                activations.append(x)
                print(x.shape)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised

        # Perform relevance propagation
        for i, layer in enumerate(self.clrp_layers):
            x = activations.pop(0)
            relevance = layer.forward(x, relevance)
            print(x.shape)

        for i, layer in enumerate(self.flrp_layers):
            x = activations.pop(0)
            relevance = layer.forward(x, relevance)
            print(x.shape)

        return relevance.permute(0, 2, 3, 1).sum(dim=-1).squeeze().detach().cpu()
