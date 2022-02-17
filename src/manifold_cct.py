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


class ManifoldCCT(nn.Module):
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
                 dropout=0.0,
                 attention_dropout=0.0,
                 stochastic_depth=0.0,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 attention_type='riem',
                 ln_attention=True,
                 *args, **kwargs):
        super(ManifoldCCT, self).__init__()
        print(attention_type, '\n\n\n\n\n\n\n\n\n')
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)
        # for p in self.tokenizer.parameters():
        #     p.requires_grad=False

        # self.project = nn.Sequential(nn.Linear(126, embedding_dim) )
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
            attention_type=attention_type,
            ln_attention=ln_attention
        )

    def forward(self, x):

        x = self.tokenizer(x)

        return self.classifier(x)


def _manifold_cct(arch, pretrained, progress,
                  num_layers, num_heads, mlp_ratio, embedding_dim,
                  kernel_size=3, stride=None, padding=None,
                  *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = ManifoldCCT(num_layers=num_layers,
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


def manifold_cct_2(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                         *args, **kwargs)


def manifold_cct_4(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                         *args, **kwargs)


def manifold_cct_6(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                         *args, **kwargs)


def manifold_cct_7(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                         *args, **kwargs)


def manifold_cct_14(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                         *args, **kwargs)


def manifold_cct_nano_(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=12, num_heads=4, mlp_ratio=4, embedding_dim=128,
                         *args, **kwargs)


def manifold_cct_tiny_(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=12, num_heads=4, mlp_ratio=4, embedding_dim=192,
                         *args, **kwargs)


def manifold_cct_small_(arch, pretrained, progress, *args, **kwargs):
    return _manifold_cct(arch, pretrained, progress, num_layers=12, num_heads=8, mlp_ratio=3, embedding_dim=384,
                         *args, **kwargs)


@register_model
def manifold_cct_2_3x2_32(pretrained=False, progress=False,
                          img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return manifold_cct_2('manifold_cct_2_3x2_32', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_2_3x2_32_sine(pretrained=False, progress=False,
                               img_size=32, positional_embedding='sine', num_classes=10,
                               *args, **kwargs):
    return manifold_cct_2('manifold_cct_2_3x2_32_sine', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_4_3x2_32(pretrained=False, progress=False,
                          img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return manifold_cct_4('manifold_cct_4_3x2_32', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_4_3x2_32_sine(pretrained=False, progress=False,
                               img_size=32, positional_embedding='sine', num_classes=10,
                               *args, **kwargs):
    return manifold_cct_4('manifold_cct_4_3x2_32_sine', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_6_3x1_32(pretrained=False, progress=False,
                          img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return manifold_cct_6('manifold_cct_6_3x1_32', pretrained, progress,
                          kernel_size=3, n_conv_layers=1,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_6_3x1_32_sine(pretrained=False, progress=False,
                               img_size=32, positional_embedding='sine', num_classes=10,
                               *args, **kwargs):
    return manifold_cct_6('manifold_cct_6_3x1_32_sine', pretrained, progress,
                          kernel_size=3, n_conv_layers=1,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_6_3x2_32(pretrained=False, progress=False,
                          img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return manifold_cct_6('manifold_cct_6_3x2_32', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_6_3x2_32_sine(pretrained=False, progress=False,
                               img_size=32, positional_embedding='sine', num_classes=10,
                               *args, **kwargs):
    return manifold_cct_6('manifold_cct_6_3x2_32_sine', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_3x1_32(pretrained=False, progress=False,
                          img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_3x1_32', pretrained, progress,
                          kernel_size=3, n_conv_layers=1,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_3x1_32_sine(pretrained=False, progress=False,
                               img_size=32, positional_embedding='sine', num_classes=10,
                               *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_3x1_32_sine', pretrained, progress,
                          kernel_size=3, n_conv_layers=1,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_3x1_32_c100(pretrained=False, progress=False,
                               img_size=32, positional_embedding='learnable', num_classes=100,
                               *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_3x1_32_c100', pretrained, progress,
                          kernel_size=3, n_conv_layers=1,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_3x1_32_sine_c100(pretrained=False, progress=False,
                                    img_size=32, positional_embedding='sine', num_classes=100,
                                    *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_3x1_32_sine_c100', pretrained, progress,
                          kernel_size=3, n_conv_layers=1,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_3x2_32(pretrained=False, progress=False,
                          img_size=32, positional_embedding='learnable', num_classes=10,
                          *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_3x2_32', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_3x2_32_sine(pretrained=False, progress=False,
                               img_size=32, positional_embedding='sine', num_classes=10,
                               *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_3x2_32_sine', pretrained, progress,
                          kernel_size=3, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_7x2_224(pretrained=False, progress=False,
                           img_size=224, positional_embedding='learnable', num_classes=102,
                           *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_7x2_224', pretrained, progress,
                          kernel_size=7, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_7_7x2_224_sine(pretrained=False, progress=False,
                                img_size=224, positional_embedding='sine', num_classes=102,
                                *args, **kwargs):
    return manifold_cct_7('manifold_cct_7_7x2_224_sine', pretrained, progress,
                          kernel_size=7, n_conv_layers=2,
                          img_size=img_size, positional_embedding=positional_embedding,
                          num_classes=num_classes,
                          *args, **kwargs)


@register_model
def manifold_cct_14_7x2_224(pretrained=False, progress=False,
                            img_size=224, positional_embedding='learnable', num_classes=1000,
                            *args, **kwargs):
    return manifold_cct_14('manifold_cct_14_7x2_224', pretrained, progress,
                           kernel_size=7, n_conv_layers=2,
                           img_size=img_size, positional_embedding=positional_embedding,
                           num_classes=num_classes,
                           *args, **kwargs)


@register_model
def manifold_cct_nano(pretrained=False, progress=False,
                      img_size=224, positional_embedding='learnable', num_classes=1000,
                      *args, **kwargs):
    return manifold_cct_nano_('manifold_cct_nano', pretrained, progress,
                              kernel_size=7, n_conv_layers=2,
                              img_size=img_size, positional_embedding=positional_embedding,
                              num_classes=num_classes,
                              *args, **kwargs)


@register_model
def manifold_cct_tiny(pretrained=False, progress=False,
                      img_size=224, positional_embedding='learnable', num_classes=1000,
                      *args, **kwargs):
    return manifold_cct_tiny_('manifold_cct_tiny', pretrained, progress,
                              kernel_size=7, n_conv_layers=2,
                              img_size=img_size, positional_embedding=positional_embedding,
                              num_classes=num_classes,
                              *args, **kwargs)


@register_model
def manifold_cct_14_7x2_384(pretrained=False, progress=False,
                            img_size=384, positional_embedding='learnable', num_classes=1000,
                            *args, **kwargs):
    return manifold_cct_14('manifold_cct_14_7x2_384', pretrained, progress,
                           kernel_size=7, n_conv_layers=2,
                           img_size=img_size, positional_embedding=positional_embedding,
                           num_classes=num_classes,
                           *args, **kwargs)


@register_model
def manifold_cct_14_7x2_384_fl(pretrained=False, progress=False,
                               img_size=384, positional_embedding='learnable', num_classes=102,
                               *args, **kwargs):
    return manifold_cct_14('manifold_cct_14_7x2_384_fl', pretrained, progress,
                           kernel_size=7, n_conv_layers=2,
                           img_size=img_size, positional_embedding=positional_embedding,
                           num_classes=num_classes,
                           *args, **kwargs)
