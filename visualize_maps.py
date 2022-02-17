#!/usr/bin/env python3
# This is a slightly modified version of timm's training script
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
from src import *
import torch
import yaml
from timm.models import create_model, resume_checkpoint
from timm.utils import *

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
# try:
#     if getattr(torch.cuda.amp, 'autocast') is not None:
#         has_native_amp = True
# except AttributeError:
#     pass

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments')

checkpoint_dict = {
    'riem': '/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results'
             '/cifar10/20220125-151023-manifold_vit_6_4_32-32/model_best.pth.tar',
    'all'  : '/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results'
             '/cifar10/20220125-222928-manifold_vit_6_4_32-32/model_best.pth.tar',
    'gm'   : '/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results'
             '/cifar10/20220201-110749-manifold_vit_6_4_32-32/model_best.pth.tar',
'e':'/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/euclidean/model_best.pth'}

sample_images = ['./data/CIFAR-10-images-master/val/horse/0010.jpg',
                 './data/CIFAR-10-images-master/val/ship/0045.jpg',
                 "./data/CIFAR-10-images-master/val/automobile/0000.jpg",
                 "./data/CIFAR-10-images-master/val/bird/0003.jpg",
                 './data/CIFAR-10-images-master/val/dog/0233.jpg',
                # './data/CIFAR-10-images-master/val/ship/0107.jpg'
                 ]

# manifold_vit_6_4_32
# vit_6_4_32
def arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Dataset / Model parameters
    parser.add_argument('--resume', default=checkpoint_dict['e'], type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')

    parser.add_argument('--attention_type', default='gm', type=str, metavar='ATT',
                        help='Type of attention to use', choices=('self', 'gm', 'riem', 'all'))

    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')
    parser.add_argument('--model', default='vit_6_4_32', type=str, metavar='MODEL',
                        help='Name of model to train (default: "countception"')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')

    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if '
                             'empty')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                        help='ratio of validation batch size to training batch size (default: 1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    return parser


def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    # print(remaining)
    # print(args_config)
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    # print(args)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)  # / 255
    cam = 0.5 * heatmap + np.float32(img)
    print(img.max(), heatmap.max())
    cam = cam / np.max(cam)
    return heatmap, cam


def main():
    parser = arguments()
    setup_default_logging()
    args, args_text = _parse_args(config_parser, parser)

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint, attention_type=args.attention_type, return_map=True)

    print(model)

    if args.resume:
        resume_checkpoint(
            model, args.resume,
            optimizer=None,
            loss_scaler=None,
            log_info=args.local_rank == 0)

    labels = dict(enumerate(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']))

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    im = Image.open(sample_images[-1]
                    )
    x = transform(im)

    logits, att_mat = model(x.unsqueeze(0),return_attention=True)

    att_mat = torch.stack([att_mat  ]).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    print(v.size())
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    hm, cam = show_cam_on_image(np.array(im), mask)
    mask2 = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    print(mask.shape, im.size)
    result = (mask / 255.0 + im).astype("uint8")
    #
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    im = cv2.resize(np.array(im),(224,224))
    cam = cv2.resize(np.array(cam),(224,224))
    hm = cv2.resize(np.array(hm/255.0),(224,224))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Overlay')
    ax3.set_title('Attention Map ')
    _ = ax1.imshow(im)
    _ = ax2.imshow(cam)
    _ = ax3.imshow(hm  )

    probs = torch.nn.Softmax(dim=-1)(logits)
    top5 = torch.argsort(probs, dim=-1, descending=True)
    print("Prediction Label and Attention Map!\n")
    for idx in top5[0, :5]:
        print(f'{probs[0, idx.item()] :.5f} : {labels[idx.item()]} ', end='')
    save_path = args.resume.rsplit('/',1)[0]
    print(save_path)
    cv2.imwrite(save_path +'/overlay_ship.jpg', cv2.cvtColor(np.uint8(cam *255 ),
                                                                                             cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path +'/map_ship.jpg',
                cv2.cvtColor(np.uint8(hm*255), cv2.COLOR_RGB2BGR))

    # axs[0, 1].plot(x, y, 'tab:orange')
    # axs[0, 1].set_title('Axis [0, 1]')
    # axs[1, 0].plot(x, -y, 'tab:green')
    # axs[1, 0].set_title('Axis [1, 0]')
    # axs[1, 1].plot(x, -y, 'tab:red')
    # axs[1, 1].set_title('Axis [1, 1]')
    # plt.savefig(args.resume.replace('model_best.pth.tar','overlay.jpg'))
    # print('MAX ',cam.max())#cv2.cvtColor(, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(args.resume.replace('model_best.pth.tar','overlay.jpg'),cv2.cvtColor(np.uint8(cam*255),
    # cv2.COLOR_RGB2BGR))
    # cv2.imwrite(args.resume.replace('model_best.pth.tar', 'map.jpg'),
    #             cv2.cvtColor(np.uint8(hm), cv2.COLOR_RGB2BGR))
    # for i, v in enumerate(joint_attentions):
    #     # Attention from the output token to the input space.
    #     mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    #     mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    #     result = (mask * im).astype("uint8")
    #
    #     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    #     ax1.set_title('Original')
    #     ax2.set_title('Attention Map_%d Layer' % (i + 1))
    #     _ = ax1.imshow(im)
    #     _ = ax2.imshow(hm/255.0)
    #
    plt.show()




def sublpot_image_patches():
    im = Image.open(
        sample_images[0])
    im = np.array(im)
    print(im.shape)
    im = cv2.resize(np.array(im), (256, 256))
    #patches = np.reshape(im,(16,64,64,3) )
    patches = torch.from_numpy(im).view(-1,64,64,3).numpy()
    print(patches.shape)
    fig, axs = plt.subplots(1, 16 ,  sharex=True, sharey=True,squeeze=True,gridspec_kw={'wspace':0, 'hspace':0},)
    fig.tight_layout()
    #im = cv2.resize(np.array(im), (224, 224))
    for i in range(4):
        for j in range(4):
            axs[j + 4*i].set_xticks([])
            axs[j + 4*i].set_yticks([])
            axs[ j + 4*i].imshow(im[i*64:(i+1)*64,j*64:(j+1)*64,:])
            #axs[i, j].set_title('Image')
    # axs[0, 0].set_xticks([])
    # axs[0, 0].set_yticks([])
    # axs[0, 0].imshow(im)
    # axs[0, 0].set_title('Image')

    plt.show()

def sublpot_attention():
    """
    Plot attention scores on the image
    """
    im = Image.open(
        sample_images[0])
    im_gm = Image.open(
        "/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results"
        "/cifar10/20220201-110749-manifold_vit_6_4_32-32/overlay_horse.jpg")
    im_all = Image.open(
        "/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results"
        "/cifar10/20220125-222928-manifold_vit_6_4_32-32/overlay_horse.jpg")
    im_spd = Image.open('/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/'
                        'paper_results/cifar10/20220125-151023-manifold_vit_6_4_32-32/overlay_horse.jpg')
    im_e = Image.open('/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/euclidean/overlay_horse.jpg')

    fig, axs = plt.subplots(3, 5,  sharex=True, sharey=True,squeeze=True,gridspec_kw={'wspace':0, 'hspace':0},)
    fig.tight_layout()
    im = cv2.resize(np.array(im), (224, 224))
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].imshow(im)
    #axs[0, 0].set_title('Image')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    axs[0, 1].imshow(im_e)
    #axs[0, 1].set_title('E')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    axs[0, 2].imshow(im_spd)
    #axs[0, 2].set_title('E_SPD')
    axs[0, 3].set_xticks([])
    axs[0, 3].set_yticks([])
    axs[0, 3].imshow(im_gm)
    #axs[0, 3].set_title('E_G')
    axs[0, 4].set_xticks([])
    axs[0, 4].set_yticks([])
    axs[0, 4].imshow(im_all)
    #axs[0, 4].set_title('E_SPD_G')




    ##################################################33

    im = Image.open(
        sample_images[-1])
    im_gm = Image.open(
        "/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results"
        "/cifar10/20220201-110749-manifold_vit_6_4_32-32/overlay_airplane.jpg")
    im_all = Image.open(
        "/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results"
        "/cifar10/20220125-222928-manifold_vit_6_4_32-32/overlay_airplane.jpg")
    im_spd = Image.open('/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/'
                        'paper_results/cifar10/20220125-151023-manifold_vit_6_4_32-32/overlay_airplane.jpg')
    im_e = Image.open('/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/euclidean/overlay_airplane.jpg')
    # fig, axs = plt.subplots(2, 4, figsize=(32, 32))
    # fig.tight_layout()


    im = cv2.resize(np.array(im), (224, 224))
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].imshow(im)
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].imshow(im_e)
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].imshow(im_spd)

    axs[1, 3].set_xticks([])
    axs[1, 3].set_yticks([])
    axs[1, 3].imshow(im_gm)

    axs[1, 4].set_xticks([])
    axs[1, 4].set_yticks([])
    axs[1, 4].imshow(im_all)


    ##################################################33

    im = Image.open(
        sample_images[2])
    im_gm = Image.open(
        "/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results"
        "/cifar10/20220201-110749-manifold_vit_6_4_32-32/overlay_automobile.jpg")
    im_all = Image.open(
        "/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/paper_results"
        "/cifar10/20220125-222928-manifold_vit_6_4_32-32/overlay_automobile.jpg")
    im_spd = Image.open('/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/'
                        'paper_results/cifar10/20220125-151023-manifold_vit_6_4_32-32/overlay_automobile.jpg')
    # fig, axs = plt.subplots(2, 4, figsize=(32, 32))
    # fig.tight_layout()
    im_e = Image.open('/home/iliask/Desktop/ilias/QCONPASS/Object_detection_research/Compact-Transformers/output/euclidean/overlay_auto.jpg')

    im = cv2.resize(np.array(im), (224, 224))
    axs[2, 0].set_xticks([])
    axs[2, 0].set_yticks([])
    axs[2, 0].imshow(im)
    axs[2, 1].set_xticks([])
    axs[2, 1].set_yticks([])
    axs[2, 1].imshow(im_e)
    axs[2, 2].set_xticks([])
    axs[2, 2].set_yticks([])
    axs[2, 2].imshow(im_spd)

    axs[2, 3].set_xticks([])
    axs[2, 3].set_yticks([])
    axs[2, 3].imshow(im_gm)

    axs[2, 4].set_xticks([])
    axs[2, 4].set_yticks([])
    axs[2, 4].imshow(im_all)

    plt.show()

#sublpot_image_patches()
#sublpot_attention()
if __name__ == '__main__':
    sublpot_attention()