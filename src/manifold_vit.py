import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .utils.helpers import pe_check
from .utils.manifold_model import ManifoldformerClassifier
from .utils.tokenizer import Tokenizer

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
    'cct_7_3x1_32'          :
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth',
    'cct_7_3x1_32_sine'     :
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained'
        '/cct_7_3x1_32_sine_cifar10_5000epochs.pth',
    'cct_7_3x1_32_c100'     :
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs'
        '.pth',
    'cct_7_3x1_32_sine_c100':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained'
        '/cct_7_3x1_32_sine_cifar100_5000epochs.pth',
    'cct_7_7x2_224_sine'    :
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth',
    'cct_14_7x2_224'        :
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'cct_14_7x2_384'        :
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    'cct_14_7x2_384_fl'     :
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class ManifoldViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=7,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 attention_type='all',
                 ln_attention=True,
                 return_map=True,
                 *args, **kwargs):
        super(ManifoldViT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = ManifoldformerClassifier(
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
            positional_embedding=positional_embedding,
            attention_type=attention_type,
            ln_attention=ln_attention,
            return_map=return_map
        )

    def forward(self, x,return_attention=False):
        x = self.tokenizer(x)
        return self.classifier(x,return_attention=return_attention)


def _manifold_vit(arch, pretrained, progress,
                  num_layers, num_heads, mlp_ratio, embedding_dim,
                  kernel_size=3, stride=None, padding=None,
                  *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = ManifoldViT(num_layers=num_layers,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        embedding_dim=embedding_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        *args, **kwargs)

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            state_dict = pe_check(model, state_dict)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
    return model


def manifold_vit_2(*args, **kwargs):
    return _manifold_vit(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                         *args, **kwargs)


def manifold_vit_4(*args, **kwargs):
    return _manifold_vit(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                         *args, **kwargs)


def manifold_vit_6(*args, **kwargs):
    return _manifold_vit(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                         *args, **kwargs)


def manifold_vit_7(*args, **kwargs):
    return _manifold_vit(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                         *args, **kwargs)


@register_model
def manifold_vit_2_4_32(pretrained=False, progress=False,
                        img_size=32, positional_embedding='learnable', num_classes=10,
                        *args, **kwargs):
    return manifold_vit_2('manifold_vit_2_4_32', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_2_4_32_sine(pretrained=False, progress=False,
                             img_size=32, positional_embedding='sine', num_classes=10,
                             *args, **kwargs):
    return manifold_vit_2('manifold_vit_2_4_32_sine', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_4_4_32(pretrained=False, progress=False,
                        img_size=32, positional_embedding='learnable', num_classes=10,
                        *args, **kwargs):
    return manifold_vit_4('manifold_vit_4_4_32', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_4_4_32_sine(pretrained=False, progress=False,
                             img_size=32, positional_embedding='sine', num_classes=10,
                             *args, **kwargs):
    return manifold_vit_4('manifold_vit_4_4_32_sine', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_6_4_32(pretrained=False, progress=False,
                        img_size=32, positional_embedding='learnable', num_classes=10,
                        *args, **kwargs):
    return manifold_vit_6('manifold_vit_6_4_32', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_6_4_32_sine(pretrained=False, progress=False,
                             img_size=32, positional_embedding='sine', num_classes=10,
                             *args, **kwargs):
    return manifold_vit_6('manifold_vit_6_4_32_sine', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_7_4_32(pretrained=False, progress=False,
                        img_size=32, positional_embedding='learnable', num_classes=10,
                        *args, **kwargs):
    return manifold_vit_7('manifold_vit_7_4_32', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_7_4_32_sine(pretrained=False, progress=False,
                             img_size=32, positional_embedding='sine', num_classes=10,
                             *args, **kwargs):
    return manifold_vit_7('manifold_vit_7_4_32_sine', pretrained, progress,
                          kernel_size=4,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_vit_nano_12_p16(pretrained=False, progress=False,
                             img_size=224, positional_embedding='learnable', num_classes=10,
                             *args, **kwargs):
    return _manifold_vit('manifold_vit_nano_12_p16', pretrained, progress,
                         kernel_size=16,
                         img_size=img_size, positional_embedding=positional_embedding,
                         num_layers=12, num_heads=4, mlp_ratio=4, embedding_dim=128,
                         num_classes=num_classes,
                         *args, **kwargs)


@register_model
def manifold_vit_tiny_12_p16(pretrained=False, progress=False,
                             img_size=224, positional_embedding='learnable', num_classes=10,
                             *args, **kwargs):
    return _manifold_vit('manifold_vit_tiny_12_p16', pretrained, progress,
                         kernel_size=16,
                         img_size=img_size, positional_embedding=positional_embedding,
                         num_layers=12, num_heads=4, mlp_ratio=4, embedding_dim=192,
                         num_classes=num_classes,
                         *args, **kwargs)


@register_model
def manifold_vit_small_12_p16(pretrained=False, progress=False,
                              img_size=224, positional_embedding='learnable', num_classes=10,
                              *args, **kwargs):
    return _manifold_vit('manifold_vit_small_12_p16', pretrained, progress,
                         kernel_size=16,
                         img_size=img_size, positional_embedding=positional_embedding,
                         num_layers=12, num_heads=8, mlp_ratio=4, embedding_dim=384,
                         num_classes=num_classes,
                         *args, **kwargs)
