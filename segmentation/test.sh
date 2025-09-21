kill -9 $(lsof -t /dev/nvidia*)
sleep 1s

# large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 . dist_test.sh configs/cityscapes/linear_dinov2_splithead_large_896_20k_cityscapes.py ./work_dirs/linear_dinov2-large_splithead_cityscapes25k_tuned10layers_frozen8layers_every3layers_start2/iter_20000.pth 8 --eval mIoU
