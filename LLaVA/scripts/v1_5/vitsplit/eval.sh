MODEL_NAME="vitsplit"
FROZEN_NUM=1
TUNED_NUM=1
CKPT="llava-v1.5-7b-$MODEL_NAME-frozen$FROZEN_NUM-tuned$TUNED_NUM"

CUDA_VISIBLE_DEVICES=7 . scripts/v1_5/eval/mmbench.sh $CKPT 
CUDA_VISIBLE_DEVICES=1 . scripts/v1_5/eval/pope.sh $CKPT 
CUDA_VISIBLE_DEVICES=0 . scripts/v1_5/eval/sqa.sh $CKPT 
CUDA_VISIBLE_DEVICES=1 . scripts/v1_5/eval/mmvet.sh $CKPT 
CUDA_VISIBLE_DEVICES=2 . scripts/v1_5/eval/llavabench.sh $CKPT 
CUDA_VISIBLE_DEVICES=0 . scripts/v1_5/eval/vqav2.sh $CKPT 
CUDA_VISIBLE_DEVICES=0 . scripts/v1_5/eval/vizwiz.sh $CKPT 
CUDA_VISIBLE_DEVICES=1 . scripts/v1_5/eval/gqa.sh $CKPT 


