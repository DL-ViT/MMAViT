# Models

## Models from paper [Escaping the Big Data Paradigm with Compact Transformers](https://arxiv.org/abs/2104.05704)
**[vit]** : models based on the classical vision transformer with class token
**[Lightweight ViT]** : Different from [ViT](https://arxiv.org/abs/2010.11929) we show that <i>an image 
is <b>not always</b> worth 16x16 words</i> and the image patch size matters.
Transformers are not in fact ''data-hungry,'' as the authors proposed, and
smaller patching can be used to train efficiently on smaller datasets.

**[cvt]** : Compact Vision Transformers better utilize information with Sequence Pooling post 
encoder, eliminating the need for the class token while achieving better
accuracy.

## Our research models with Riemmanian and Grassmanian Manifold
#### early_fusion_image_grassmanian_vit 

Early fusion of image patches and observability matrices as input to a vanilla transformer

### grassmanian_cct 


### grassmanian_vit

Grassmanian  Transformer with attention that adopts SVD and QR decompositions to extract observability and orthogonal matrices,
while the attention is calculated from distance on grasmmanian manifold

```
python train.py -c configs/datasets/dataset.yml --model gm_vit_6_4_32   /dataset_path
```


### img_gm_vit   

Late fusion of output of the two transformers, vanilla-Transformer and Grassmanian Transformer

```
python train.py -c configs/datasets/dataset.yml --model img_gm_vit_6_4_32   /dataset_path
```

### img_riem_vit 

Late fusion of output of the two transformers, vanilla-Transformer and Riemmanian Transformer


```
python train.py -c configs/datasets/dataset.yml --model img_riem_vit_6_4_32   /dataset_path
```

###  riemmanian_vit 

Riemmanian Transformer with attention that adopts covariance matrices and distance on riemmanian manifold



```
python train.py -c configs/datasets/dataset.yml --model riem_vit_6_4_32   /dataset_path
```

### img_gm_riem_vit  

```
python train.py -c configs/datasets/dataset.yml --model img_gm_riem_vit_6_4_32   /dataset_path
```


### gm_riem_vit  

```
python train.py -c configs/datasets/dataset.yml --model gm_riem_vit_6_4_32   /dataset_path
```



### manifold_cct  


To train manifold_cct with:

- Riemmanian+Self attention
```
python train.py -c configs/datasets/dataset.yml --model manifold_ctt_7_3x2_32   --attention_type riem  /datapath
```

- Riemmanian+Grassmanian+Self attention
```
python train.py -c configs/datasets/dataset.yml --model manifold_ctt_7_3x2_32  --attention_type all  /datapath
```

### manifold_vit  

To train manifold_vit with:

- Riemmanian+Self attention
```
python train.py -c configs/datasets/dataset.yml --model manifold_vit_x_y_32 - --attention_type riem  /datapath
```

- Riemmanian+Grassmanian+Self attention
```
python train.py -c configs/datasets/dataset.yml --model manifold_vit_x_y_32 --gpu 0 --attention_type all  /datapath
```