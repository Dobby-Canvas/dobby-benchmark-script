import copy
import functools
import inspect
import itertools
import logging
import math
import threading
import warnings
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.ao.nn.quantized as nnq
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, TextualInversionLoaderMixin
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization import PlaceholderObserver, QConfig
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig import (_activation_is_memoryless, _add_module_to_qconfig_obs_ctr,
                                           default_dynamic_qconfig, float16_dynamic_qconfig,
                                           float_qparams_weight_only_qconfig, float_qparams_weight_only_qconfig_4bit)
from torch.ao.quantization.quantization_mappings import (
    _get_special_act_post_process, _has_special_act_post_process, get_default_dynamic_quant_module_mappings,
    get_default_qat_module_mappings, get_default_qconfig_propagation_list, get_default_static_quant_module_mappings,
    get_default_static_quant_reference_module_mappings, no_observer_set)
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
from torch.ao.quantization.utils import (get_qparam_dict, has_no_children_ignoring_parametrizations)
from torch.nn.utils.parametrize import type_before_parametrizations
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from diffusers.loaders import StableDiffusionLoraLoaderMixin
except ImportError:
    StableDiffusionLoraLoaderMixin = None
import mixdq_extension._C
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import \
    StableDiffusionPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor

from .sd15_quant_config import a8_mixed_precision_config, w8_uniform_config

logger = logging.getLogger(__name__)

######################################################################################################
# quant ops

quantize_per_tensor = mixdq_extension._C.quantize_per_tensor_to_int8


def qconv2d(
    input_int,
    weight_int,
    weight_scale,
    input_scale,
    input_zp,
    scale,
    weight_sum_by_input_channels,
    bias0,
    bias=None,
    stride=1,
    padding=0,
):
    dilation = 1
    return mixdq_extension._C.qconv2d_w8_a8_ohalf(input_int, weight_int, weight_scale, input_scale, input_zp, scale,
                                                  weight_sum_by_input_channels, bias0, bias, stride, padding, dilation)


qlinear = mixdq_extension._C.qlinear_w8_a8_ohalf

# quant ops
######################################################################################################

######################################################################################################
# PyTorch quantization infrastructure (adapted from torch.ao.quantization)

_DEFAULT_CUSTOM_CONFIG_DICT = {
    'float_to_observed_custom_module_class': {
        nn.LSTM: nn.quantizable.LSTM,
        nn.MultiheadAttention: nn.quantizable.MultiheadAttention,
    },
    'observed_to_quantized_custom_module_class': {
        nn.quantizable.LSTM: nn.quantized.LSTM,
        nn.quantizable.MultiheadAttention: nn.quantized.MultiheadAttention,
    }
}

# SD1.5 UNet up_blocks conv_shortcut split dimensions (in order of traversal).
#
# In diffusers UNet the concatenation is: cat([hidden_states, res_hidden_states], dim=1),
# so split = hidden_states.shape[1] (the *current* feature channel count).
#
# Block channel flow (block_out_channels = (320, 640, 1280, 1280)):
#   UB0 (UpBlock2D,        1280ch output): current=1280 for all 3 resnets       → [1280, 1280, 1280]
#   UB1 (CrossAttn,        1280ch output): current=1280 for all 3 resnets       → [1280, 1280, 1280]
#   UB2 (CrossAttn,  640ch output): r0 current=1280, r1/r2 current=640          → [1280, 640, 640]
#   UB3 (CrossAttn,  320ch output): r0 current=640,  r1/r2 current=320          → [640,  320, 320]
_SPLIT = [1280, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 640, 320, 320]

_NUM = 0


def get_default_custom_config_dict():
    return _DEFAULT_CUSTOM_CONFIG_DICT


def _propagate_qconfig_helper(module, qconfig_dict, qconfig_parent=None, prefix='', prepare_custom_config_dict=None):
    module_qconfig = qconfig_dict.get(type_before_parametrizations(module), qconfig_parent)
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    module_qconfig = getattr(module, 'qconfig', module_qconfig)

    torch.ao.quantization.qconfig._assert_valid_qconfig(module_qconfig, module)

    qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(module_qconfig, module)
    module.qconfig = qconfig_with_device_check

    for name, child in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        if prepare_custom_config_dict is None or not (
                name in prepare_custom_config_dict.get("non_traceable_module_name", [])
                or type(child) in prepare_custom_config_dict.get("non_traceable_module_class", [])):
            _propagate_qconfig_helper(child, qconfig_dict, qconfig_with_device_check, module_prefix)


def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None):
    if qconfig_dict is None:
        qconfig_dict = {}
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    _propagate_qconfig_helper(module, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)


def _observer_forward_hook(self, input, output):
    return self.activation_post_process(output)


def _observer_forward_pre_hook(self, input):
    return self.activation_post_process(input[0])


def _register_activation_post_process_hook(module, pre_hook=False):
    assert hasattr(module, 'activation_post_process')
    if pre_hook:
        module.register_forward_pre_hook(_observer_forward_pre_hook, prepend=True)
    else:
        module.register_forward_hook(_observer_forward_hook, prepend=True)


def _get_unique_devices_(module):
    return {p.device for p in module.parameters()} | {p.device for p in module.buffers()}


def _add_observer_(module,
                   qconfig_propagation_list=None,
                   non_leaf_module_list=None,
                   device=None,
                   custom_module_class_mapping=None):
    if qconfig_propagation_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()

    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}

    if device is None:
        devices = _get_unique_devices_(module)
        assert len(devices) <= 1
        device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device, special_act_post_process=None):
        activation = qconfig.activation() if special_act_post_process is None \
            else special_act_post_process()
        if device is not None:
            activation.to(device)
        return activation

    def needs_observation(m):
        return hasattr(m, 'qconfig') and m.qconfig is not None

    def insert_activation_post_process(m, special_act_post_process=None):
        if needs_observation(m) and not isinstance(m, DeQuantStub):
            m.add_module('activation_post_process',
                         get_activation_post_process(m.qconfig, device, special_act_post_process))
            _register_activation_post_process_hook(m, pre_hook=_activation_is_memoryless(m.qconfig))

    for name, child in module.named_children():
        if type_before_parametrizations(child) in [nn.Dropout]:
            continue
        elif issubclass(type_before_parametrizations(child), (nnq.FloatFunctional, nnq.QFunctional)):
            if needs_observation(child):
                assert hasattr(child, "activation_post_process")
                child.activation_post_process = get_activation_post_process(child.qconfig, device)
        elif isinstance(child, _FusedModule):
            if needs_observation(child):
                insert_activation_post_process(child)
        elif non_leaf_module_list is not None and \
                type_before_parametrizations(child) in non_leaf_module_list:
            if needs_observation(child):
                insert_activation_post_process(child)
        elif _has_special_act_post_process(child):
            special_act_post_process = _get_special_act_post_process(child)
            insert_activation_post_process(child, special_act_post_process)
        elif needs_observation(child) and \
                type_before_parametrizations(child) in custom_module_class_mapping:
            observed_child = custom_module_class_mapping[type_before_parametrizations(child)].from_float(child)
            setattr(module, name, observed_child)
            if custom_module_class_mapping[type_before_parametrizations(child)] \
                    not in no_observer_set():
                insert_activation_post_process(observed_child)
        else:
            _add_observer_(child, qconfig_propagation_list, non_leaf_module_list, device, custom_module_class_mapping)

    if has_no_children_ignoring_parametrizations(module) \
            and not isinstance(module, torch.nn.Sequential) \
            and type_before_parametrizations(module) in qconfig_propagation_list:
        insert_activation_post_process(module)


def _remove_activation_post_process(module):
    if hasattr(module, 'activation_post_process') and \
            _is_activation_post_process(module.activation_post_process):
        delattr(module, 'activation_post_process')

    def remove_hooks(pre_hook=False):
        hook_map = module._forward_pre_hooks if pre_hook else module._forward_hooks
        observer_hook = _observer_forward_pre_hook if pre_hook else _observer_forward_hook
        handle_ids_to_remove = {hid for hid, fn in hook_map.items() if fn is observer_hook}
        for hid in handle_ids_to_remove:
            hook_map.pop(hid)

    remove_hooks(pre_hook=True)
    remove_hooks(pre_hook=False)


def _remove_qconfig(module):
    for child in module.children():
        _remove_qconfig(child)
    if hasattr(module, "qconfig"):
        del module.qconfig
    _remove_activation_post_process(module)


def prepare(model, inplace=False, allow_list=None, observer_non_leaf_module_list=None, prepare_custom_config_dict=None):
    torch._C._log_api_usage_once("quantization_api.quantize.prepare")
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = prepare_custom_config_dict.get("float_to_observed_custom_module_class", {})

    if not inplace:
        model = copy.deepcopy(model)

    qconfig_propagation_list = allow_list
    if allow_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()
    propagate_qconfig_(model, qconfig_dict=None)

    if not any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules()):
        warnings.warn("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")

    _add_observer_(model,
                   qconfig_propagation_list,
                   observer_non_leaf_module_list,
                   custom_module_class_mapping=custom_module_class_mapping)
    return model


def swap_module(mod, mapping, custom_module_class_mapping, ckpt=None):
    global _NUM
    new_mod = mod
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        swapped = False
        if type_before_parametrizations(mod) in custom_module_class_mapping:
            new_mod = custom_module_class_mapping[type_before_parametrizations(mod)].from_observed(mod)
            swapped = True
        elif type_before_parametrizations(mod) in mapping:
            qmod = mapping[type_before_parametrizations(mod)]
            if hasattr(qmod, '_IS_REFERENCE') and qmod._IS_REFERENCE:
                assert mod.qconfig is not None
                weight_post_process = mod.qconfig.weight()
                weight_post_process(mod.weight)
                weight_qparams = get_qparam_dict(weight_post_process)
                if 'up_blocks' in mod.module_name and 'conv_shortcut' in mod.module_name:
                    _split = _SPLIT[_NUM]
                    _NUM = _NUM + 1
                else:
                    _split = 0
                new_mod = qmod.from_float(mod, weight_qparams, split=_split)
            else:
                if 'up_blocks' in mod.module_name and 'conv_shortcut' in mod.module_name:
                    _split = _SPLIT[_NUM]
                    _NUM = _NUM + 1

                else:
                    _split = 0
                new_mod = qmod.from_float(mod, split=_split, ckpt=ckpt)
            swapped = True

        if swapped:
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            for hook_fn in mod._forward_hooks.values():
                if hook_fn is not _observer_forward_hook:
                    new_mod.register_forward_hook(hook_fn)

            devices = _get_unique_devices_(mod)
            assert len(devices) <= 1
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                new_mod.to(device)
    return new_mod


def _convert(module, mapping=None, inplace=False, is_reference=False, convert_custom_config_dict=None, ckpt=None):
    if mapping is None:
        mapping = get_default_static_quant_reference_module_mappings() if is_reference \
            else get_default_static_quant_module_mappings()
    if convert_custom_config_dict is None:
        convert_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = convert_custom_config_dict.get("observed_to_quantized_custom_module_class", {})

    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        if not isinstance(mod, _FusedModule) and \
                type_before_parametrizations(mod) not in custom_module_class_mapping:
            _convert(mod, mapping, True, is_reference, convert_custom_config_dict, ckpt=ckpt)
        reassign[name] = swap_module(mod, mapping, custom_module_class_mapping, ckpt=ckpt)

    for key, value in reassign.items():
        module._modules[key] = value

    return module


def convert(module,
            mapping=None,
            inplace=False,
            remove_qconfig=True,
            is_reference=False,
            convert_custom_config_dict=None,
            ckpt=None):
    torch._C._log_api_usage_once("quantization_api.quantize.convert")
    if not inplace:
        module = copy.deepcopy(module)
    _convert(module,
             mapping,
             inplace=True,
             is_reference=is_reference,
             convert_custom_config_dict=convert_custom_config_dict,
             ckpt=ckpt)
    if remove_qconfig:
        _remove_qconfig(module)
    return module


# PyTorch quantization infrastructure
######################################################################################################

######################################################################################################
# MixDQ quantization utilities


def quantize_per_tensor_uint4(input: torch.Tensor, scale, zero_point):
    scale = scale.view(-1, *([1] * (len(input.shape) - 1)))
    zero_point = zero_point.view(-1, *([1] * (len(input.shape) - 1)))
    scale_inv = 1.0 / scale
    int_repr = torch.clamp(torch.round(input * scale_inv) + zero_point, 0, 15).to(torch.uint8)
    if len(input.shape) >= 4:
        assert input.shape[1] % 2 == 0
        return (int_repr[:, ::2, ...] << 4 | int_repr[:, 1::2, ...])
    assert input.shape[-1] % 2 == 0
    return (int_repr[..., ::2] << 4 | int_repr[..., 1::2])


def unpack_uint4(input):
    shape = input.shape
    if len(shape) >= 4:
        packed_dim = 2
        new_shape = (input.shape[0], input.shape[1] * 2, *input.shape[2:])
    else:
        packed_dim = -1
        new_shape = (*input.shape[:-1], input.shape[-1] * 2)
    first_elements = (input >> 4).to(torch.uint8)
    second_elements = (input & 0b1111).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=packed_dim).view(new_shape)


def dequantize_per_tensor_uint4(input, scale, zero_point):
    scale = scale.view(-1, *([1] * (len(input.shape) - 1)))
    zero_point = zero_point.view(-1, *([1] * (len(input.shape) - 1)))
    input = unpack_uint4(input)
    return (input.view(torch.uint8).to(torch.float32) - zero_point) * scale


dtype_to_bw = {
    torch.quint8: 8,
    torch.quint4x2: 4,
    torch.quint2x4: 2,
    torch.float16: 16,
}


class QParam(
        namedtuple("QParam", ["qscheme", "dtype", "scales", "zero_points", "axis"],
                   defaults=[torch.per_tensor_affine, torch.quint8, 1.0, 0.0, 0])):

    @property
    def zp_float(self):
        return self.scales * self.zero_points


def get_quant_para(ckpt, n_bit, module_name, quant_type, split=0, device=None):
    """Parse quantization parameters from a checkpoint.

    Supports two checkpoint formats:
    - qdiff PTQ format: ckpt[key] = [OrderedDict({'delta': scale, 'zero_point': zp}), ...]
      where zero_point is stored in uint8 space [0, 255]; converted to signed [-128, 127].
    - MixDQ SDXL format: ckpt[key] = {'delta_list': [...], 'zero_point_list': [...]}
    """
    suffix = '.weight_quantizer' if quant_type == 'weight' else '.act_quantizer'
    key = module_name + suffix
    assert key in ckpt, f"Key '{key}' not found in checkpoint"

    def _parse_entry(raw):
        if isinstance(raw, list):
            # qdiff PTQ: [OrderedDict({'delta': tensor, 'zero_point': tensor}), ...]
            entry = raw[0]
            scales = entry['delta'].reshape(-1).float()
            # qdiff stores zero_points in uint8 space [0, 255]; shift to signed [-128, 127]
            zero_points = (entry['zero_point'].reshape(-1) - 128).float()
        elif isinstance(raw, dict) and 'delta_list' in raw:
            # MixDQ SDXL: {'delta_list': [...], 'zero_point_list': [...]}
            bit_idx = int(math.log2(n_bit) - 1)
            scales = raw['delta_list'][bit_idx]
            zero_points = raw['zero_point_list'][bit_idx]
            if quant_type == 'act':
                zero_points = (zero_points - 128).float()
        else:
            raise ValueError(f"Unknown checkpoint format for key '{key}'")
        return scales, zero_points

    scales, zero_points = _parse_entry(ckpt[key])

    if split == 0:
        return scales.to(device), zero_points.to(device), None, None

    # For split layers, try to get separate params for each half.
    # qdiff checkpoints don't have '_0' keys; reuse the same params in that case.
    key_0 = key + '_0'
    if key_0 in ckpt:
        scales_0, zero_points_0 = _parse_entry(ckpt[key_0])
    else:
        scales_0, zero_points_0 = scales, zero_points

    return (scales.to(device), zero_points.to(device), scales_0.to(device), zero_points_0.to(device))


def create_qparams_from_dtype(dtype,
                              device,
                              is_channel_wise=False,
                              num_kernels=None,
                              ckpt=None,
                              module_name=None,
                              bit_width=0,
                              quant_type=None,
                              split=0):
    if dtype == torch.float16:
        return None

    elif dtype in [torch.qint8, torch.quint8, torch.quint4x2]:
        if quant_type == 'weight':
            scales, zero_points, scales_0, zero_points_0 = get_quant_para(ckpt,
                                                                          bit_width,
                                                                          module_name,
                                                                          quant_type='weight',
                                                                          split=split,
                                                                          device=device)
        elif quant_type == 'act':
            scales, zero_points, scales_0, zero_points_0 = get_quant_para(ckpt,
                                                                          bit_width,
                                                                          module_name,
                                                                          quant_type='act',
                                                                          split=split,
                                                                          device=device)
    else:
        raise ValueError(f"Unsupported quantize dtype {dtype}")

    if is_channel_wise:
        assert num_kernels is not None
        qparam = QParam(qscheme=torch.per_channel_affine, scales=scales, zero_points=zero_points, dtype=dtype, axis=0)
        qparam_0 = QParam(
            qscheme=torch.per_channel_affine, scales=scales_0, zero_points=zero_points_0, dtype=dtype,
            axis=0) if split > 0 else None
    else:
        qparam = QParam(qscheme=torch.per_tensor_affine, scales=scales, zero_points=zero_points, dtype=dtype)
        qparam_0 = QParam(qscheme=torch.per_tensor_affine, scales=scales_0, zero_points=zero_points_0,
                          dtype=dtype) if split > 0 else None

    return qparam, qparam_0


def dequantize_to_float16(x: torch.Tensor, qparams: QParam):
    if x.dtype == torch.float16:
        return x
    if x.dtype in [torch.quint8, torch.qint8]:
        return x.dequantize().to(torch.float16)
    elif x.dtype in [torch.int8]:
        scale = (qparams.scales.view(-1, *([1] * (len(x.shape) - 1)))).cuda()
        zero_points = (qparams.zero_points.view(-1, *([1] * (len(x.shape) - 1)))).cuda()
        return scale * (x - zero_points)
    assert x.dtype == torch.uint8
    return dequantize_per_tensor_uint4(x, qparams.scales.to(x.device),
                                       qparams.zero_points.to(x.device)).to(torch.float16)


# MixDQ quantization utilities
######################################################################################################

######################################################################################################
# MixDQ quantized modules


class QuantizedConv2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 groups=1,
                 bias=True,
                 device=None,
                 w_qparams=None,
                 w_qparams_0=None,
                 a_qparams=None,
                 a_qparams_0=None,
                 module_name=None,
                 split=0) -> None:
        super().__init__()

        self.module_name = module_name
        self.split = split
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.valid_for_acceleration = (
            w_qparams is not None and a_qparams is not None and w_qparams.dtype in [torch.qint8, torch.quint8]
            and a_qparams.dtype in [torch.qint8, torch.quint8] and w_qparams.qscheme == torch.per_channel_affine
            and a_qparams.qscheme == torch.per_tensor_affine and torch.all(w_qparams.zero_points == 0.0).item() and
            (split == 0 or
             (w_qparams_0 is not None and a_qparams_0 is not None and w_qparams_0.dtype in [torch.qint8, torch.quint8]
              and a_qparams_0.dtype in [torch.qint8, torch.quint8] and w_qparams_0.qscheme == torch.per_channel_affine
              and a_qparams_0.qscheme == torch.per_tensor_affine and torch.all(w_qparams_0.zero_points == 0.0).item()))
            and (len(set(self.stride)) == 1 and len(set(self.padding)) == 1 and len(set(self.dilation)) == 1
                 and self.dilation[0] == 1 and self.groups == 1))
        if self.valid_for_acceleration and (self.in_channels % 4 != 0 or self.out_channels % 4 != 0):
            logging.warning(f"Conv2d layer with in_channels={self.in_channels} and "
                            f"out_channels={self.out_channels} cannot use quantized kernel "
                            "due to misalignment. Falling back to FP kernels")
            self.valid_for_acceleration = False

        # int8 storage is available whenever we have per-channel weight qparams,
        # even for asymmetric quantization (non-zero zero_point) that can't use CUDA kernels.
        self.valid_for_int8_storage = (w_qparams is not None and w_qparams.dtype in [torch.qint8, torch.quint8]
                                       and w_qparams.qscheme == torch.per_channel_affine)

        if self.valid_for_acceleration:
            self.register_buffer("weight_scales", w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", w_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales", a_qparams.scales.to(device).float())
            self.register_buffer("act_zero_points", a_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales_inv", 1 / self.act_scales)
            if self.split != 0:
                self.register_buffer("weight_scales_0", w_qparams_0.scales.to(device).float())
                self.register_buffer("weight_zero_points_0", w_qparams_0.zero_points.to(device).float())
                self.register_buffer("act_scales_0", a_qparams_0.scales.to(device).float())
                self.register_buffer("act_zero_points_0", a_qparams_0.zero_points.to(device).float())
                self.register_buffer("act_scales_inv_0", 1 / self.act_scales_0)
        elif self.valid_for_int8_storage:
            # Store quantization params for the dequantize-then-compute path.
            self.register_buffer("weight_scales", w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", w_qparams.zero_points.to(device).float())
            if split != 0 and w_qparams_0 is not None:
                self.register_buffer("weight_scales_0", w_qparams_0.scales.to(device).float())
                self.register_buffer("weight_zero_points_0", w_qparams_0.zero_points.to(device).float())

    @classmethod
    def from_float(cls, float_mod, split=0, ckpt=None):
        assert hasattr(float_mod, 'qconfig') and isinstance(float_mod.qconfig, QConfig)
        weight_process = float_mod.qconfig.weight()
        w_dtype = weight_process.dtype
        num_kernels = float_mod.weight.shape[0]
        device = float_mod.weight.device

        w_qparams, w_qparams_0 = create_qparams_from_dtype(dtype=w_dtype,
                                                           device=device,
                                                           is_channel_wise=True,
                                                           num_kernels=num_kernels,
                                                           ckpt=ckpt,
                                                           module_name=float_mod.module_name,
                                                           quant_type='weight',
                                                           bit_width=float_mod.w_bit,
                                                           split=split)

        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype

        if hasattr(float_mod, 'a_bit'):
            a_qparams, a_qparams_0 = create_qparams_from_dtype(dtype=act_dtype,
                                                               device=device,
                                                               is_channel_wise=False,
                                                               num_kernels=num_kernels,
                                                               ckpt=ckpt,
                                                               module_name=float_mod.module_name,
                                                               quant_type='act',
                                                               bit_width=float_mod.a_bit,
                                                               split=split)
        else:
            a_qparams = None
            a_qparams_0 = None

        new_mod = cls(float_mod.in_channels,
                      float_mod.out_channels,
                      float_mod.kernel_size,
                      float_mod.stride,
                      float_mod.padding,
                      float_mod.dilation,
                      float_mod.groups,
                      float_mod.bias is not None,
                      device=float_mod.weight.device,
                      w_qparams=w_qparams,
                      w_qparams_0=w_qparams_0,
                      a_qparams=a_qparams,
                      a_qparams_0=a_qparams_0,
                      module_name=float_mod.module_name,
                      split=split)

        weight = float_mod.weight.detach()

        def _quantize_weight_to_int8(w, scales, zero_points):
            """Manually quantize weight to int8 using per-channel params."""
            s = scales.view(-1, *([1] * (w.dim() - 1)))
            zp = zero_points.view(-1, *([1] * (w.dim() - 1)))
            return torch.clamp(torch.round(w.float() / s) + zp, -128, 127).to(torch.int8)

        if split == 0:
            if new_mod.valid_for_acceleration:
                weight_int = torch.quantize_per_channel(weight.float(),
                                                        new_mod.weight_scales,
                                                        new_mod.weight_zero_points,
                                                        axis=w_qparams.axis,
                                                        dtype=w_qparams.dtype).int_repr()
                new_mod.register_buffer("weight_int", weight_int)

                if float_mod.padding[0] == 0:
                    weight_sum = weight_int.float().sum(dim=[1, 2, 3])
                    new_mod.register_buffer("bias0", weight_sum * new_mod.act_zero_points)
                    new_mod.weight_sum_by_input_channels = None
                else:
                    weight_sum_by_input_channels = weight_int.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels", weight_sum_by_input_channels)
                    new_mod.bias0 = None
                new_mod.register_buffer("scale", new_mod.weight_scales * new_mod.act_scales)
            elif new_mod.valid_for_int8_storage:
                # Asymmetric quantization: store as int8 for memory savings; dequantize at runtime.
                weight_int = _quantize_weight_to_int8(weight, new_mod.weight_scales, new_mod.weight_zero_points)
                new_mod.register_buffer("weight_int", weight_int)
            else:
                new_mod.register_buffer("weight", weight)

            if float_mod.bias is not None:
                new_mod.register_buffer("bias", float_mod.bias.detach())
            else:
                new_mod.bias = None

        elif split > 0:
            if new_mod.valid_for_acceleration:
                weight_int = torch.quantize_per_channel(weight[:, :split, ...].float(),
                                                        new_mod.weight_scales,
                                                        new_mod.weight_zero_points,
                                                        axis=w_qparams.axis,
                                                        dtype=w_qparams.dtype).int_repr()
                weight_int_0 = torch.quantize_per_channel(weight[:, split:, ...].float(),
                                                          new_mod.weight_scales_0,
                                                          new_mod.weight_zero_points_0,
                                                          axis=w_qparams_0.axis,
                                                          dtype=w_qparams_0.dtype).int_repr()

                new_mod.register_buffer("weight_int", weight_int)
                new_mod.register_buffer("weight_int_0", weight_int_0)

                if float_mod.padding[0] == 0:
                    weight_sum = weight_int.float().sum(dim=[1, 2, 3])
                    new_mod.register_buffer("bias0", weight_sum * new_mod.act_zero_points)
                    weight_sum_0 = weight_int_0.float().sum(dim=[1, 2, 3])
                    new_mod.register_buffer("bias0_0", weight_sum_0 * new_mod.act_zero_points_0)
                    new_mod.weight_sum_by_input_channels = None
                    new_mod.weight_sum_by_input_channels_0 = None
                else:
                    weight_sum_by_ic = weight_int.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels", weight_sum_by_ic)
                    weight_sum_by_ic_0 = weight_int_0.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels_0", weight_sum_by_ic_0)
                    new_mod.bias0 = None
                    new_mod.bias0_0 = None
                new_mod.register_buffer("scale", new_mod.weight_scales * new_mod.act_scales)
                new_mod.register_buffer("scale_0", new_mod.weight_scales_0 * new_mod.act_scales_0)
            elif new_mod.valid_for_int8_storage:
                scales_0 = getattr(new_mod, 'weight_scales_0', new_mod.weight_scales)
                zps_0 = getattr(new_mod, 'weight_zero_points_0', new_mod.weight_zero_points)
                weight_int = _quantize_weight_to_int8(weight[:, :split, ...], new_mod.weight_scales,
                                                      new_mod.weight_zero_points)
                weight_int_0 = _quantize_weight_to_int8(weight[:, split:, ...], scales_0, zps_0)
                new_mod.register_buffer("weight_int", weight_int)
                new_mod.register_buffer("weight_int_0", weight_int_0)
                if not hasattr(new_mod, 'weight_scales_0'):
                    new_mod.register_buffer("weight_scales_0", scales_0)
                    new_mod.register_buffer("weight_zero_points_0", zps_0)
            else:
                new_mod.register_buffer("weight", weight)

            if float_mod.bias is not None:
                new_mod.register_buffer("bias", float_mod.bias.detach())
            else:
                new_mod.bias = None

        return new_mod

    def _get_name(self):
        if self.valid_for_acceleration:
            return "QuantizedConv2dW8A8"
        if self.valid_for_int8_storage:
            return "QuantizedConv2dInt8Dequant"
        return "QuantizedConv2dFPFallback"

    def _dequantize_weight(self, weight_int, scales, zero_points):
        s = scales.view(-1, *([1] * (weight_int.dim() - 1)))
        zp = zero_points.view(-1, *([1] * (weight_int.dim() - 1)))
        return (weight_int.float() - zp) * s

    def forward_fallback(self, x: torch.Tensor):
        """Called when valid_for_acceleration=True but input is not fp16."""
        weight_recovered = self._dequantize_weight(self.weight_int, self.weight_scales,
                                                   self.weight_zero_points).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        if self.split == 0:
            return F.conv2d(x, weight_recovered, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            weight_0_recovered = self._dequantize_weight(self.weight_int_0, self.weight_scales_0,
                                                         self.weight_zero_points_0).to(x.dtype)
            output = F.conv2d(x[:, :self.split, :, :], weight_recovered, bias, self.stride, self.padding, self.dilation,
                              self.groups)
            output_0 = F.conv2d(x[:, self.split:, :, :], weight_0_recovered, None, self.stride, self.padding,
                                self.dilation, self.groups)
            return output + output_0

    def forward_dequant(self, x: torch.Tensor):
        """Dequantize int8 weight then compute in FP (for asymmetric quantization)."""
        weight_recovered = self._dequantize_weight(self.weight_int, self.weight_scales,
                                                   self.weight_zero_points).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        if self.split == 0:
            return F.conv2d(x, weight_recovered, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            weight_0_recovered = self._dequantize_weight(self.weight_int_0, self.weight_scales_0,
                                                         self.weight_zero_points_0).to(x.dtype)
            output = F.conv2d(x[:, :self.split, :, :], weight_recovered, bias, self.stride, self.padding, self.dilation,
                              self.groups)
            output_0 = F.conv2d(x[:, self.split:, :, :], weight_0_recovered, None, self.stride, self.padding,
                                self.dilation, self.groups)
            return output + output_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.valid_for_acceleration:
            if self.valid_for_int8_storage:
                return self.forward_dequant(x)
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if not x.dtype == torch.float16:
            return self.forward_fallback(x)

        if self.split == 0:
            x_int = quantize_per_tensor(x, self.act_scales_inv, self.act_zero_points)
            return qconv2d(x_int, self.weight_int, self.weight_scales, self.act_scales, self.act_zero_points,
                           self.scale, self.weight_sum_by_input_channels, self.bias0, self.bias, self.stride[0],
                           self.padding[0])
        else:
            x_int = quantize_per_tensor(x[:, :self.split, :, :], self.act_scales_inv, self.act_zero_points)
            x_int_0 = quantize_per_tensor(x[:, self.split:, :, :], self.act_scales_inv_0, self.act_zero_points_0)
            output = qconv2d(x_int, self.weight_int, self.weight_scales, self.act_scales, self.act_zero_points,
                             self.scale, self.weight_sum_by_input_channels, self.bias0, self.bias, self.stride[0],
                             self.padding[0])
            output_0 = qconv2d(x_int_0, self.weight_int_0, self.weight_scales_0, self.act_scales_0,
                               self.act_zero_points_0, self.scale_0, self.weight_sum_by_input_channels_0, self.bias0_0,
                               None, self.stride[0], self.padding[0])
            return output + output_0


class QuantizedLinear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 w_qparams=None,
                 a_qparams=None,
                 module_name=None) -> None:
        super().__init__()
        self.module_name = module_name
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.valid_for_acceleration = (w_qparams is not None and a_qparams is not None
                                       and w_qparams.dtype in [torch.qint8, torch.quint8]
                                       and a_qparams.dtype in [torch.qint8, torch.quint8]
                                       and w_qparams.qscheme == torch.per_channel_affine
                                       and a_qparams.qscheme == torch.per_tensor_affine
                                       and torch.all(w_qparams.zero_points == 0.0).item())
        if self.valid_for_acceleration and (self.in_features % 4 != 0 or self.out_features % 4 != 0):
            logging.warning(f"Linear layer with in_features={self.in_features} and "
                            f"out_features={self.out_features} cannot use quantized kernel "
                            "due to misalignment. Falling back to FP kernels")
            self.valid_for_acceleration = False

        self.valid_for_int8_storage = (w_qparams is not None and w_qparams.dtype in [torch.qint8, torch.quint8]
                                       and w_qparams.qscheme == torch.per_channel_affine)

        if self.valid_for_acceleration:
            self.register_buffer("weight_scales", w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", w_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales", a_qparams.scales.to(device).float())
            self.register_buffer("act_zero_points", a_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales_inv", 1 / self.act_scales)
        elif self.valid_for_int8_storage:
            self.register_buffer("weight_scales", w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", w_qparams.zero_points.to(device).float())

    @classmethod
    def from_float(cls, float_mod, split=0, ckpt=None):
        assert hasattr(float_mod, 'qconfig') and isinstance(float_mod.qconfig, QConfig)
        weight_process = float_mod.qconfig.weight()
        w_dtype = weight_process.dtype
        num_kernels = float_mod.weight.shape[0]
        device = float_mod.weight.device

        w_qparams, _ = create_qparams_from_dtype(dtype=w_dtype,
                                                 device=device,
                                                 is_channel_wise=True,
                                                 num_kernels=num_kernels,
                                                 ckpt=ckpt,
                                                 module_name=float_mod.module_name,
                                                 quant_type='weight',
                                                 bit_width=float_mod.w_bit,
                                                 split=split)

        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype

        if hasattr(float_mod, 'a_bit'):
            a_qparams, _ = create_qparams_from_dtype(dtype=act_dtype,
                                                     device=device,
                                                     is_channel_wise=False,
                                                     num_kernels=num_kernels,
                                                     ckpt=ckpt,
                                                     module_name=float_mod.module_name,
                                                     quant_type='act',
                                                     bit_width=float_mod.a_bit,
                                                     split=split)
        else:
            a_qparams = None

        new_mod = cls(float_mod.in_features,
                      float_mod.out_features,
                      float_mod.bias is not None,
                      device=float_mod.weight.device,
                      w_qparams=w_qparams,
                      a_qparams=a_qparams,
                      module_name=float_mod.module_name)

        weight = float_mod.weight.detach()

        # SD1.5 cross-attention uses attn2 (same as SDXL); carry bos if present
        if 'attn2' in float_mod.module_name:
            if 'to_k' in float_mod.module_name or 'to_v' in float_mod.module_name:
                new_mod.bos = getattr(float_mod, 'bos', False)
                if new_mod.bos and hasattr(float_mod, 'bos_pre_computed'):
                    new_mod.register_buffer("bos_pre_computed", float_mod.bos_pre_computed)

        if new_mod.valid_for_acceleration:
            weight_int = torch.quantize_per_channel(weight.float(),
                                                    new_mod.weight_scales,
                                                    new_mod.weight_zero_points,
                                                    axis=w_qparams.axis,
                                                    dtype=w_qparams.dtype).int_repr()
            new_mod.register_buffer("weight_int", weight_int)

            weight_sum_by_ic = weight_int.float().sum(dim=1)
            new_mod.register_buffer("weight_sum_by_input_channels", weight_sum_by_ic)
            new_mod.register_buffer("scale", new_mod.weight_scales * new_mod.act_scales)
            new_mod.register_buffer("bias0", weight_sum_by_ic * new_mod.act_zero_points)
        elif new_mod.valid_for_int8_storage:
            s = new_mod.weight_scales.view(-1, 1)
            zp = new_mod.weight_zero_points.view(-1, 1)
            weight_int = torch.clamp(torch.round(weight.float() / s) + zp, -128, 127).to(torch.int8)
            new_mod.register_buffer("weight_int", weight_int)
        else:
            new_mod.register_buffer("weight", weight)

        if float_mod.bias is not None:
            new_mod.register_buffer("bias", float_mod.bias.detach())
        else:
            new_mod.bias = None

        return new_mod

    def _get_name(self):
        if self.valid_for_acceleration:
            return "QuantizedLinearW8A8"
        if self.valid_for_int8_storage:
            return "QuantizedLinearInt8Dequant"
        return "QuantizedLinearFPFallback"

    def forward_fallback(self, x):
        """Called when valid_for_acceleration=True but input is not fp16."""
        zp = self.weight_zero_points.view(-1, 1)
        weight_recovered = ((self.weight_int.float() - zp) * self.weight_scales[:, None]).to(x.dtype)
        return F.linear(x, weight_recovered, self.bias.to(x.dtype) if self.bias is not None else None)

    def forward_dequant(self, x):
        """Dequantize int8 weight then compute in FP (for asymmetric quantization)."""
        zp = self.weight_zero_points.view(-1, 1)
        weight_recovered = ((self.weight_int.float() - zp) * self.weight_scales[:, None]).to(x.dtype)
        return F.linear(x, weight_recovered, self.bias.to(x.dtype) if self.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.valid_for_acceleration:
            if self.valid_for_int8_storage:
                return self.forward_dequant(x)
            return F.linear(x, self.weight, self.bias)

        if not x.dtype == torch.float16:
            return self.forward_fallback(x)

        if not hasattr(self, 'bos') or not self.bos:
            x_int = quantize_per_tensor(x, self.act_scales_inv, self.act_zero_points)
            return qlinear(x_int, self.weight_int, self.weight_scales, self.act_scales, self.act_zero_points,
                           self.weight_sum_by_input_channels, self.scale, self.bias0, self.bias)
        else:
            x_except_first_token = quantize_per_tensor(x[:, 1:, :], self.act_scales_inv, self.act_zero_points)
            out_except_first_token = qlinear(x_except_first_token, self.weight_int, self.weight_scales, self.act_scales,
                                             self.act_zero_points, self.weight_sum_by_input_channels, self.scale,
                                             self.bias0, self.bias)
            out_first_token = self.bos_pre_computed.expand(x.shape[0], -1, -1)
            return torch.cat([out_first_token, out_except_first_token], dim=1)


# MixDQ quantized modules
######################################################################################################

######################################################################################################
# Pipeline helpers


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


# Pipeline helpers
######################################################################################################

######################################################################################################
# Quantization wiring


def filter_mod_name_prefix(mod_name):
    if 'model.' in mod_name:
        pos = mod_name.index('model.')
        mod_name = mod_name[pos + 6:]
    return mod_name


def register_qconfig_from_input_files(unet, w_bit=8, a_bit=None, bos=True, bos_dict=None):
    bw_to_dtype = {
        8: torch.qint8,
        4: torch.quint4x2,
        2: torch.quint4x2,
    }

    if w_bit == 8:
        mod_name_to_weight_width = w8_uniform_config
    else:
        raise RuntimeError("Only INT8 weight quantization is supported")

    mod_name_to_weight_width = {filter_mod_name_prefix(k): v for k, v in mod_name_to_weight_width.items()}
    remaining_weight = dict(mod_name_to_weight_width)

    for name, mod in unet.named_modules():
        if name in mod_name_to_weight_width:
            assert not hasattr(mod, 'qconfig')
            w_bitwidth = mod_name_to_weight_width[name]
            w_dtype = bw_to_dtype[w_bitwidth]
            mod.qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16),
                                  weight=PlaceholderObserver.with_args(dtype=w_dtype))

            mod.module_name = name
            mod.w_bit = w_bitwidth

            if 'attn2' in name and ('to_k' in name or 'to_v' in name):
                mod.bos = bos
                if bos and bos_dict is not None:
                    mod.bos_pre_computed = bos_dict[name]

            del remaining_weight[name]

    if remaining_weight:
        for name in remaining_weight:
            print(f"{name} not found in UNet!")
        raise RuntimeError("Not all weight-config keys mapped to a UNet module.")

    if a_bit is None:
        return

    if a_bit == 8:
        mod_name_to_act_width = a8_mixed_precision_config
    else:
        raise RuntimeError("Only INT8 activation quantization is supported")

    mod_name_to_act_width = {filter_mod_name_prefix(k): v for k, v in mod_name_to_act_width.items()}
    remaining_act = dict(mod_name_to_act_width)

    for name, mod in unet.named_modules():
        if name in mod_name_to_act_width:
            a_bitwidth = mod_name_to_act_width[name]
            a_dtype = bw_to_dtype[a_bitwidth]
            act_preprocess = PlaceholderObserver.with_args(dtype=a_dtype)

            if hasattr(mod, 'qconfig') and mod.qconfig:
                mod.qconfig = QConfig(weight=mod.qconfig.weight, activation=act_preprocess)
            else:
                mod.qconfig = QConfig(activation=act_preprocess,
                                      weight=PlaceholderObserver.with_args(dtype=torch.float16))

            mod.a_bit = a_bitwidth
            del remaining_act[name]

    if remaining_act:
        for name in remaining_act:
            print(f"{name} not found in UNet!")
        raise RuntimeError("Not all act-config keys mapped to a UNet module.")


def convert_to_quantized(unet, ckpt):
    convert(unet, mapping={nn.Linear: QuantizedLinear, nn.Conv2d: QuantizedConv2d}, inplace=True, ckpt=ckpt)


# Quantization wiring
######################################################################################################

######################################################################################################
# SD1.5 MixDQ Pipeline


class MixDQ_SD15_Pipeline_W8A8(
        DiffusionPipeline,
        FromSingleFileMixin,
        TextualInversionLoaderMixin,
):
    """
    Stable Diffusion 1.5 pipeline with MixDQ W8A8 quantization support.

    Adapts the MixDQ quantization infrastructure (originally for SDXL-Turbo) to
    Stable Diffusion 1.5.  The main differences from the SDXL variant are:

    * Single text encoder / tokenizer (no second CLIP encoder)
    * No pooled text embeddings or SDXL conditioning embeddings
    * Default guidance scale of 7.5 instead of 0.0
    * Simpler denoising loop without SDXL-specific ``added_cond_kwargs``
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if (untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
                    and not torch.equal(text_input_ids, untruncated_ids)):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
                logger.warning("The following part of your input was truncated because CLIP can only "
                               f"handle sequences up to {self.tokenizer.model_max_length} tokens: "
                               f"{removed_text}")

            encoder_output = self.text_encoder(text_input_ids.to(device), output_hidden_states=True)

            if clip_skip is None:
                prompt_embeds = encoder_output.last_hidden_state
            else:
                prompt_embeds = encoder_output.hidden_states[-(clip_skip + 1)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_output = self.text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)

            if clip_skip is None:
                negative_prompt_embeds = uncond_output.last_hidden_state
            else:
                negative_prompt_embeds = uncond_output.hidden_states[-(clip_skip + 1)]

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    # ------------------------------------------------------------------
    # Latent / scheduler helpers
    # ------------------------------------------------------------------

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`, not both.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` must be str or list, got {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Provide either `negative_prompt` or `negative_prompt_embeds`, not both.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError("`prompt_embeds` and `negative_prompt_embeds` must have the same shape.")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"You passed {len(generator)} generators for batch size {batch_size}.")

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize_unet(
        self,
        ckpt_path: Optional[str] = None,
        bos_path: Optional[str] = None,
        w_bit: int = 8,
        a_bit: Optional[int] = None,
        bos: bool = False,
    ):
        """
        Apply MixDQ W8A8 quantization to the UNet.

        Args:
            ckpt_path: Path to a ``ckpt.pth`` quant-params checkpoint.
                       If *None*, attempts to download from the default
                       ``nics-efc/MixDQ`` HuggingFace repository (SD1.5 variant).
            bos_path:  Path to the pre-computed BOS tensor file.  Only needed
                       when ``bos=True``.
            w_bit:     Bit-width for weight quantization (default: 8).
            a_bit:     Bit-width for activation quantization (``None`` = W-only).
            bos:       Whether to enable the Block-wise Online Softmax technique.
        """
        global _NUM
        _NUM = 0

        if ckpt_path is None:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id="nics-efc/MixDQ",
                filename="sd15_quant_para_wsym_fp16.pt",
            )
        else:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=ckpt_path,
                filename="quantization/ckpt.pth",
            )

        ckpt = torch.load(ckpt_path, map_location='cpu')

        bos_dict = None
        if bos:
            if bos_path is None:
                from huggingface_hub import hf_hub_download
                bos_path = hf_hub_download(
                    repo_id="nics-efc/MixDQ",
                    filename="sd15_bos_pre_computed.pt",
                )
            bos_dict = torch.load(bos_path, map_location='cpu')

        register_qconfig_from_input_files(self.unet, w_bit=w_bit, a_bit=a_bit, bos=bos, bos_dict=bos_dict)
        convert_to_quantized(self.unet, ckpt)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated. "
                "Consider using `callback_on_step_end`.",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated. "
                "Consider using `callback_on_step_end`.",
            )

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds,
                          callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        device = self._execution_device

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width,
                                       prompt_embeds.dtype, device, generator, latents)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


# SD1.5 MixDQ Pipeline
######################################################################################################
