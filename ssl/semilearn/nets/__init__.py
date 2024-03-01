# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet50, resnet34, resnet18
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
# from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
# from .bert import bert_base_cased, bert_base_uncased
# from .wave2vecv2 import wave2vecv2_base
# from .hubert import hubert_base
from .cifar10_models import resnet32
from .mobilenetv2 import mobilenet_v2
from .vit_pytorch import vit_b_16
from .resnet_big import SupConResNet34 as supconresnet34
from .gbcnet import GbcNet
from .radformer.radformer import radformer
from .deit import deit_base_patch16_224
from .visiontransformer import vit_b_16
