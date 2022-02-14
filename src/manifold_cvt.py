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


class ManifoldCVT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 attention_type='all',
                 *args, **kwargs):
        super(ManifoldCVT, self).__init__()

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
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            attention_type=attention_type
        )

    def forward(self, x):
        x = self.tokenizer(x)

        # print(x.shape, om.shape)
        # print(om.shape)
        # x  = self.tokenizer(x)sss
        # om = self.project(om)

        return self.classifier(x)



def _manifold_cvt(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=4, *args, **kwargs):
    model = ManifoldCVT(num_layers=num_layers,
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


def manifold_cvt_2(*args, **kwargs):
    return _manifold_cvt(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def manifold_cvt_4(*args, **kwargs):
    return _manifold_cvt(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def manifold_cvt_6(*args, **kwargs):
    return _manifold_cvt(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def manifold_cvt_7(*args, **kwargs):
    return _manifold_cvt(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def manifold_cvt_8(*args, **kwargs):
    return _manifold_cvt(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


@register_model
def manifold_cvt_2_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return manifold_cvt_2('manifold_cvt_2_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def manifold_cvt_2_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return manifold_cvt_2('manifold_cvt_2_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def manifold_cvt_4_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return manifold_cvt_4('manifold_cvt_4_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def manifold_cvt_4_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return manifold_cvt_4('manifold_cvt_4_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def manifold_cvt_6_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return manifold_cvt_6('manifold_cvt_6_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def manifold_cvt_6_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return manifold_cvt_6('manifold_cvt_6_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def manifold_cvt_7_4_32(pretrained=False, progress=False,
               img_size=32, positional_embedding='learnable', num_classes=10,
               *args, **kwargs):
    return manifold_cvt_7('manifold_cvt_7_4_32', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def manifold_cvt_7_4_32_sine(pretrained=False, progress=False,
                    img_size=32, positional_embedding='sine', num_classes=10,
                    *args, **kwargs):
    return manifold_cvt_7('manifold_cvt_7_4_32_sine', pretrained, progress,
                 kernel_size=4,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)
