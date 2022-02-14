import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .utils.grassmanian_models import GrassmanianformerClassifier, ObsMatrixTokenizer
from .utils.helpers import pe_check
from .utils.tokenizer import Tokenizer

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
}


class GrassmanianViTLite(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(GrassmanianViTLite, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)
        self.m = 3
        self.lds_order = 8
        self.om_layer = nn.Sequential(nn.LayerNorm(embedding_dim),
                                      ObsMatrixTokenizer(image_size=img_size, patch_size=kernel_size, m=self.m,
                                                         lds_size=self.lds_order),
                                      nn.Linear(self.m * self.lds_order ** 2, embedding_dim))

        self.classifier = GrassmanianformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x,return_attention=False):
        x = self.tokenizer(x)

        om = self.om_layer(x)

        return self.classifier(om,return_attention=return_attention)


def _grassmanian_vit_lite(arch, pretrained, progress,
                          num_layers, num_heads, mlp_ratio, embedding_dim,
                          kernel_size=4, *args, **kwargs):
    model = GrassmanianViTLite(num_layers=num_layers,
                               num_heads=num_heads,
                               mlp_ratio=mlp_ratio,
                               embedding_dim=embedding_dim,
                               kernel_size=kernel_size,
                               *args, **kwargs)

    if pretrained and arch in model_urls:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict = pe_check(model, state_dict)
        model.load_state_dict(state_dict)
    return model


def grassmanian_vit_2(*args, **kwargs):
    return _grassmanian_vit_lite(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                                 *args, **kwargs)


def grassmanian_vit_4(*args, **kwargs):
    return _grassmanian_vit_lite(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                                 *args, **kwargs)


def grassmanian_vit_6(*args, **kwargs):
    return _grassmanian_vit_lite(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                                 *args, **kwargs)


def grassmanian_vit_7(*args, **kwargs):
    return _grassmanian_vit_lite(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                                 *args, **kwargs)


@register_model
def grassmanian_vit_2_4_32(pretrained=False, progress=False,
                           img_size=32, positional_embedding='learnable', num_classes=10,
                           *args, **kwargs):
    return grassmanian_vit_2('grassmanian_vit_2_4_32', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)


@register_model
def grassmanian_vit_2_4_32_sine(pretrained=False, progress=False,
                                img_size=32, positional_embedding='sine', num_classes=10,
                                *args, **kwargs):
    return grassmanian_vit_2('grassmanian_vit_2_4_32_sine', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)


@register_model
def grassmanian_vit_4_4_32(pretrained=False, progress=False,
                           img_size=32, positional_embedding='learnable', num_classes=10,
                           *args, **kwargs):
    return grassmanian_vit_4('grassmanian_vit_4_4_32', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)


@register_model
def grassmanian_vit_4_4_32_sine(pretrained=False, progress=False,
                                img_size=32, positional_embedding='sine', num_classes=10,
                                *args, **kwargs):
    return grassmanian_vit_4('grassmanian_vit_4_4_32_sine', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)


@register_model
def grassmanian_vit_6_4_32(pretrained=False, progress=False,
                           img_size=32, positional_embedding='learnable', num_classes=10,
                           *args, **kwargs):
    return grassmanian_vit_6('grassmanian_vit_6_4_32', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)


@register_model
def grassmanian_vit_6_4_32_sine(pretrained=False, progress=False,
                                img_size=32, positional_embedding='sine', num_classes=10,
                                *args, **kwargs):
    return grassmanian_vit_6('grassmanian_vit_6_4_32_sine', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)


@register_model
def grassmanian_vit_7_4_32(pretrained=False, progress=False,
                           img_size=32, positional_embedding='learnable', num_classes=10,
                           *args, **kwargs):
    return grassmanian_vit_7('grassmanian_vit_7_4_32', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)


@register_model
def grassmanian_vit_7_4_32_sine(pretrained=False, progress=False,
                                img_size=32, positional_embedding='sine', num_classes=10,
                                *args, **kwargs):
    return grassmanian_vit_7('grassmanian_vit_7_4_32_sine', pretrained, progress,
                             kernel_size=4,
                             img_size=img_size, positional_embedding=positional_embedding,
                             num_classes=num_classes,
                             *args, **kwargs)
