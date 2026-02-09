import math

import torch
import torch.nn as nn
from typing import Union


@torch.no_grad()
def _batchnorm_reset(layer: Union[nn.BatchNorm1d, nn.BatchNorm2d], reset_interpolation_factor: float) -> None:
    """Reset the weights and the running mean and variance of a batchnorm layer."""

    # Sample phi from initializer distribution
    phi_weight = torch.ones(layer.weight.data.shape).to(layer.weight.device)

    # Apply soft reset to weights
    layer.weight.data.mul_(reset_interpolation_factor)
    layer.weight.data.add_((1 - reset_interpolation_factor) * phi_weight)
    # Bias init is zero, so no need to add phi_bias
    layer.bias.data.mul_(reset_interpolation_factor)


@torch.no_grad()
def _conv_linear_kaiming_reset(layer: Union[nn.Linear, nn.Conv2d], reset_interpolation_factor: float) -> None:
    """
    Reset weights by applying a soft reset.
    θ_t = alpha*θ_t-1 + (1-alpha)*phi
    """
    # Sample phi from initializer distribution
    phi = torch.empty(layer.weight.data.shape).to(layer.weight.device)
    nn.init.kaiming_uniform_(phi)
    # Apply transformation
    layer.weight.data.mul_(reset_interpolation_factor)
    layer.weight.data.add_((1 - reset_interpolation_factor) * phi)

    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight.data)
        phi_bias = torch.empty(layer.bias.data.shape).to(layer.bias.device)
        if isinstance(layer, nn.Conv2d):
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(phi_bias, -bound, bound)
                layer.bias.data.mul_(reset_interpolation_factor)
                layer.bias.data.add_((1 - reset_interpolation_factor) * phi_bias)
        else:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(phi_bias, -bound, bound)
            layer.bias.data.mul_(reset_interpolation_factor)
            layer.bias.data.add_((1 - reset_interpolation_factor) * phi_bias)


def _check_layer_reset_eligibility(
    layer: nn.Module, embedding_layer: nn.Linear, reset_embedding: bool, reset_batchnorm: bool
) -> bool:
    """Check if a layer is eligible for soft reset."""
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)) and not layer.weight.requires_grad:
        return False
    elif reset_embedding and isinstance(layer, (nn.Linear , nn.Conv2d)):
        # Reset all layers irrespective of whether they are the embedding layer or not
        return True
    elif not reset_embedding and isinstance(layer, nn.Conv2d):
        # Convolutional layers are always reset because they can't be the embedding layer
        return True
    elif not reset_embedding and isinstance(layer, nn.Linear):
        # Linear layers are reset only if they are not the embedding layer
        return layer is not embedding_layer
    elif reset_batchnorm and isinstance(layer, (nn.BatchNorm1d , nn.BatchNorm2d)):
        return True
    else:
        return False


@torch.no_grad()
def soft_reset(
    autoencoder: Union[nn.Module, nn.ModuleList],
    reset_interpolation_factor: float,
    reset_interpolation_factor_step: float,
    reset_batchnorm: bool,
    reset_embedding: bool,
    reset_projector: bool,
    reset_convlayers: bool,
) -> None:
    candidate_layers_dict = {}
    reset_factors_dict = {}
    reset_factors_dict["fc_modules"] = reset_interpolation_factor

    # 适配BERT_USNID
    if hasattr(autoencoder, 'mlp_head'):
        candidate_layers_dict["fc_modules"] = [autoencoder.dense, autoencoder.mlp_head]
        embedding_layer = autoencoder.mlp_head
        if hasattr(autoencoder, 'classifier') and autoencoder.args.pretrain:
            candidate_layers_dict["fc_modules"].append(autoencoder.classifier)
    else:
        candidate_layers_dict["fc_modules"] = list(autoencoder.modules())
        embedding_layer = autoencoder.block[-1] if hasattr(autoencoder, 'block') else None

    if reset_projector and hasattr(autoencoder, 'projector'):
        candidate_layers_dict["projector"] = list(autoencoder.projector.modules())
        reset_factors_dict["projector"] = reset_interpolation_factor

    for key, candidate_layers in candidate_layers_dict.items():
        candidate_layers_dict[key] = [
            layer for layer in candidate_layers
            if _check_layer_reset_eligibility(layer, embedding_layer, reset_embedding, reset_batchnorm)
        ]

    print(f"Applying Soft Reset with alpha={reset_interpolation_factor}.")
    for key, layers_to_be_reset in candidate_layers_dict.items():
        reset_factor_i = reset_factors_dict[key]
        for layer in layers_to_be_reset:
            if isinstance(layer, nn.Conv2d):
                print(f"reset convlayer {key} with alpha={reset_factor_i}")
                _conv_linear_kaiming_reset(layer, reset_factor_i)
            elif isinstance(layer, nn.Linear):
                print(f"reset linear layer {key} with alpha={reset_factor_i}")
                _conv_linear_kaiming_reset(layer, reset_interpolation_factor)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                _batchnorm_reset(layer, reset_interpolation_factor)
                print(f"reset batchnorm layer {key} with alpha={reset_factor_i}")
            else:
                raise ValueError(f"Layer {layer} is not supported for soft reset.")
