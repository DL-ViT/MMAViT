#  Multi-Manifold Attention Vision Transformers



Overview of the network architecture of a Vision Transformer with its main components	

![VIT](images/vit.png)


Multi-manifold multi-head attention

![Manifold attention](images/mahsa.png)

# How to run




### Models and config files
We've updated this repository and moved the previous training script and the checkpoints associated 
with it to `examples/`. The new training script here is just the `timm` training script. We've provided
the checkpoints associated with it in the next section, and the hyperparameters are all provided in
`configs/pretrained` for models trained from scratch, and `configs/finetuned` for fine-tuned models.




# Results

## Ablation study

| Euclidean | SPD | Grassmann | Params (M) | FLOPS (G) | CIFAR-10 | CIFAR-100 |
|-----------|-----|-----------|------------|-----------|----------|-----------|
| X         |     |           | 3.19       | 0.21      | 90.94    | 69.2      |
|           | X   |           | 3.20       | 0.22      | 84.78    | 67.57     |
|           |     | X         | 3.19       | 0.22      | 83.83    | 62.36     |
| X         | X   |           | 3.60       | 0.23      | 92.77    | 75.35     |
| X         |     | X         | 3.60       | 0.24      | 92.19    | 75.53     |
|           | X   | X         | 3.60       | 0.24      | 84.21    | 67.84     |
| X         | X   | X         | 3.81       | 0.25      | 92.91    | 75.70     |


## Manifold ViTs

| Method                | Params (M) | FLOPS (G) | CIFAR-10 | CIFAR-100 | MNIST |
|-----------------------|------------|-----------|----------|-----------|-------|
| Manifold-ViT-Lite-6/4 | 3.81       | 0.25      | 92.91    | 75.70     | 99.47 |
| Manifold-CVT-6/4      | 3.78       | 0.24      | 94.63    | 77.05     | 99.42 |
| Manifold-CCT-7/3×2    | 4.54       | 0.32      | 95.28    | 79.31     | 99.56 |


### late fusion models

| Method                 | Euclidean | SPD | Grassmann | Params (M) | FLOPS (G) | CIFAR-10 | CIFAR-100 |
|------------------------|-----------|-----|-----------|------------|-----------|----------|-----------|
| gm_riem_vit_6_4_32     |           | X   | X         | 6.62       | 0.44      | 86.07    | 70.19     |
| img_gm_vit_6_4_32      | X         |     | X         | 6.62       | 0.44      | 91.68    | 73.89     |
| img_riem_vit_6_4_32    | X         | X   |           | 6.62       | 0.44      | 92.63    | 74.32     |
| img_gm_riem_vit_6_4_32 | X         | X   | X         | 9.80       | 0.67      | 92.65    | 73.97     |


Type can be read in the format `L/PxC` where `L` is the number of transformer
layers, `P` is the patch/convolution size, and `C` (CCT only) is the number of
convolutional layers.





 

## Install locally

### Requirements

Python 3.7

Our base model is in pure PyTorch and Torchvision. No extra packages are required.
Please refer to [PyTorch's Getting Started](https://pytorch.org/get-started/locally/) page for detailed instructions.

 

## Training

[timm](https://github.com/rwightman/pytorch-image-models) is recommended for image classification training 
and required for the training script provided in this repository:
### Distributed training
```shell
./dist_classification.sh $NUM_GPUS -c $CONFIG_FILE /path/to/dataset
```

You can use our training configurations provided in `configs/`:
```shell
./dist_classification.sh 2 -c configs/imagenet.yml --model manifold_cct_7_7x2_224 /home/papastrat/Desktop/imagenet
```

### Non-distributed training
```shell
python train.py -c configs/datasets/cifar10.yml --model cct_7_3x1_32 /path/to/cifar10
```
```shell
python train.py -c configs/datasets/cifar10.yml --model grassmanian_vit_6_4_32 --gpu 0 --log-wandb ./data/CIFAR10/
```


## Arguments

```commandline

    positional arguments:
      DIR                   path to dataset
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset NAME, -d NAME
                            dataset type (default: ImageFolder/ImageTar if empty)
      --train-split NAME    dataset train split (default: train)
      --val-split NAME      dataset validation split (default: validation)
      --model MODEL         Name of model to train (default: "countception"
      --attention_type ATT  Type of attention to use
      --pretrained          Start with pretrained version of specified network (if
                            avail)
      --initial-checkpoint PATH
                            Initialize model from this checkpoint (default: none)
      --resume PATH         Resume full model and optimizer state from checkpoint
                            (default: none)
      --no-resume-opt       prevent resume of optimizer state when resuming model
      --num-classes N       number of label classes (Model default if None)
      --gp POOL             Global pool type, one of (fast, avg, max, avgmax,
                            avgmaxc). Model default if None.
      --img-size N          Image patch size (default: None => model default)
      --input-size N N N N N N N N N
                            Input all image dimensions (d h w, e.g. --input-size 3
                            224 224), uses model default if empty
      --crop-pct N          Input image center crop percent (for validation only)
      --mean MEAN [MEAN ...]
                            Override mean pixel value of dataset
      --std STD [STD ...]   Override std deviation of of dataset
      --interpolation NAME  Image resize interpolation type (overrides model)
      -b N, --batch-size N  input batch size for training (default: 32)
      -vb N, --validation-batch-size-multiplier N
                            ratio of validation batch size to training batch size
                            (default: 1)
      --opt OPTIMIZER       Optimizer (default: "sgd"
      --opt-eps EPSILON     Optimizer Epsilon (default: None, use opt default)
      --opt-betas BETA [BETA ...]
                            Optimizer Betas (default: None, use opt default)
      --momentum M          Optimizer momentum (default: 0.9)
      --weight-decay WEIGHT_DECAY
                            weight decay (default: 0.0001)
      --clip-grad NORM      Clip gradient norm (default: None, no clipping)
      --clip-mode CLIP_MODE
                            Gradient clipping mode. One of ("norm", "value",
                            "agc")
      --gradient_steps GRADIENT_STEPS
      --sched SCHEDULER     LR scheduler (default: "step"
      --lr LR               learning rate (default: 0.01)
      --lr-noise pct, pct [pct, pct ...]
                            learning rate noise on/off epoch percentages
      --lr-noise-pct PERCENT
                            learning rate noise limit percent (default: 0.67)
      --lr-noise-std STDDEV
                            learning rate noise std-dev (default: 1.0)
      --lr-cycle-mul MULT   learning rate cycle len multiplier (default: 1.0)
      --lr-cycle-limit N    learning rate cycle limit
      --warmup-lr LR        warmup learning rate (default: 0.0001)
      --min-lr LR           lower lr bound for cyclic schedulers that hit 0 (1e-5)
      --epochs N            number of epochs to train (default: 2)
      --epoch-repeats N     epoch repeat multiplier (number of times to repeat
                            dataset epoch per train epoch).
      --start-epoch N       manual epoch number (useful on restarts)
      --decay-epochs N      epoch interval to decay LR
      --warmup-epochs N     epochs to warmup LR, if scheduler supports
      --cooldown-epochs N   epochs to cooldown LR at min_lr, after cyclic schedule
                            ends
      --patience-epochs N   patience epochs for Plateau LR scheduler (default: 10
      --decay-rate RATE, --dr RATE
                            LR decay rate (default: 0.1)
      --no-aug              Disable all training augmentation, override other
                            train aug args
      --scale PCT [PCT ...]
                            Random resize scale (default: 0.08 1.0)
      --ratio RATIO [RATIO ...]
                            Random resize aspect ratio (default: 0.75 1.33)
      --hflip HFLIP         Horizontal flip training aug probability
      --vflip VFLIP         Vertical flip training aug probability
      --color-jitter PCT    Color jitter factor (default: 0.4)
      --aa NAME             Use AutoAugment policy. "v0" or "original". (default:
                            None)
      --aug-splits AUG_SPLITS
                            Number of augmentation splits (default: 0, valid: 0 or
                            >=2)
      --jsd                 Enable Jensen-Shannon Divergence + CE loss. Use with
                            `--aug-splits`.
      --reprob PCT          Random erase prob (default: 0.)
      --remode REMODE       Random erase mode (default: "const")
      --recount RECOUNT     Random erase count (default: 1)
      --resplit             Do not random erase first (clean) augmentation split
      --mixup MIXUP         mixup alpha, mixup enabled if > 0. (default: 0.)
      --cutmix CUTMIX       cutmix alpha, cutmix enabled if > 0. (default: 0.)
      --cutmix-minmax CUTMIX_MINMAX [CUTMIX_MINMAX ...]
                            cutmix min/max ratio, overrides alpha and enables
                            cutmix if set (default: None)
      --mixup-prob MIXUP_PROB
                            Probability of performing mixup or cutmix when
                            either/both is enabled
      --mixup-switch-prob MIXUP_SWITCH_PROB
                            Probability of switching to cutmix when both mixup and
                            cutmix enabled
      --mixup-mode MIXUP_MODE
                            How to apply mixup/cutmix params. Per "batch", "pair",
                            or "elem"
      --mixup-off-epoch N   Turn off mixup after this epoch, disabled if 0
                            (default: 0)
      --smoothing SMOOTHING
                            Label smoothing (default: 0.1)
      --train-interpolation TRAIN_INTERPOLATION
                            Training interpolation (random, bilinear, bicubic
                            default: "random")
      --drop PCT            Dropout rate (default: 0.)
      --drop-connect PCT    Drop connect rate, DEPRECATED, use drop-path (default:
                            None)
      --drop-path PCT       Drop path rate (default: None)
      --drop-block PCT      Drop block rate (default: None)
      --bn-tf               Use Tensorflow BatchNorm defaults for models that
                            support it (default: False)
      --bn-momentum BN_MOMENTUM
                            BatchNorm momentum override (if not None)
      --bn-eps BN_EPS       BatchNorm epsilon override (if not None)
      --sync-bn             Enable NVIDIA Apex or Torch synchronized BatchNorm.
      --dist-bn DIST_BN     Distribute BatchNorm stats between nodes after each
                            epoch ("broadcast", "reduce", or "")
      --split-bn            Enable separate BN layers per augmentation split.
      --model-ema           Enable tracking moving average of model weights
      --model-ema-force-cpu
                            Force ema to be tracked on CPU, rank=0 node only.
                            Disables EMA validation.
      --model-ema-decay MODEL_EMA_DECAY
                            decay factor for model weights moving average
                            (default: 0.9998)
      --seed S              random seed (default: 42)
      --log-interval N      how many batches to wait before logging training
                            status
      --recovery-interval N
                            how many batches to wait before writing recovery
                            checkpoint
      --checkpoint-hist N   number of checkpoints to keep (default: 10)
      -j N, --workers N     how many training processes to use (default: 1)
      --save-images         save images of input bathes every log interval for
                            debugging
      --amp                 use NVIDIA Apex AMP or Native AMP for mixed precision
                            training
      --apex-amp            Use NVIDIA Apex AMP mixed precision
      --native-amp          Use Native Torch AMP mixed precision
      --channels-last       Use channels_last memory layout
      --pin-mem             Pin CPU memory in DataLoader for more efficient
                            (sometimes) transfer to GPU.
      --no-prefetcher       disable fast prefetcher
      --output PATH         path to output folder (default: none, current dir)
      --experiment NAME     name of train experiment, name of sub-folder for
                            output
      --eval-metric EVAL_METRIC
                            Best metric (default: "top1"
      --tta N               Test/inference time augmentation (oversampling)
                            factor. 0=None (default: 0)
      --local_rank LOCAL_RANK
      --gpu GPU
      --use-multi-epochs-loader
                            use the multi-epochs-loader to save time at the
                            beginning of every epoch
      --torchscript         convert model torchscript for inference
      --log-wandb           log training and validation metrics to wandb


```




## Acknowledgments

Thanks to   [SHI Lab](https://github.com/SHI-Labs) and the [timm](https://github.com/rwightman/pytorch-image-models) library from [Ross Wightman](https://github.com/rwightman)  for the awesome code for the transformers,
which I have used for my research implementations. 



# Citation

```bibtex
@article{hassani2021escaping,
	title        = {Escaping the Big Data Paradigm with Compact Transformers},
	author       = {Ali Hassani and Steven Walton and Nikhil Shah and Abulikemu Abuduweili and Jiachen Li and Humphrey Shi},
	year         = 2021,
	url          = {https://arxiv.org/abs/2104.05704},
	eprint       = {2104.05704},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```
