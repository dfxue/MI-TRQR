# train imagenet script 
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 
python -m torch.distributed.launch --nproc_per_node=4 --master_port 1267 --use_env train.py --cos_lr_T 320 --model sew_resnet34 \
-b 64 --output-dir ./logs --tb --print-freq 2048 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 \
--data-path /mnt/data/dfxue/datasets/ILSVRC/Data/CLS-LOC --tet -j 8 \
--load /mnt/data/dfxue/disk/codes-snn/LBP/imagenet/pretrained/resnet34-b627a593.pth \
