docker run --rm -it --init \
  --ipc=host \
  --volume="$PWD:/app" \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  compacttr:latest python train.py -c configs/datasets/cifar10.yml --model manifold_vit_2_4_32 ./data/CIFAR-10-images-master
