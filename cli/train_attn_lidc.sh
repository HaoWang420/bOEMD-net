python train.py \
       --workers 4 \
       --lr 0.5 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 32 \
       --test-batch-size 1 \
       --checkname attn-unet-lidc \
       --eval-interval 5 \
       --dataset lidc \
       --loss-type dice \
       --nchannels 1 \
       --model attn-unet \
       --nclass 4 \
       --task-num 0
       
