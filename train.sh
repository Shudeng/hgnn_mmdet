## single gpu
#python train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py

## multi-gpu distribute train
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py --launcher pytorch

#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29503 train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py --launcher pytorch

export PYTHONPATH=$PWD:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29709 ./tools/train.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py --launcher pytorch --work-dir gan

#CUDA_LAUNCH_BLOCKING=1 python train_mmdet.py configs/votenet/votenet_16x8_sunrgbd-3d-10class.py

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type PlainHead

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type VoteHead --distributed

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
#    --head_type VoteHead

#python train.py configs/_base_/datasets/kitti-3d-3class.py \
    #--head_type PlainHead
