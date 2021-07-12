python train.py \
    --workers 4 \
    --lr 0.01 \
    --epochs 200 \
    --gpu-ids 0,1,2,3 \
    --batch-size 4 \
    --test-batch-size 1 \
    --checkname multi-unet-prostate0 \
    --eval-interval 1 \
    --dataset uncertain-prostate \
    --loss-type dice \
    --nchannels 1 \
    --model unet \
    --nclass 3 \
    --task-num 0

python train.py \
    --workers 4 \
    --lr 0.01 \
    --epochs 200 \
    --gpu-ids 0,1,2,3 \
    --batch-size 4 \
    --test-batch-size 1 \
    --checkname multi-unet-prostate0 \
    --eval-interval 1 \
    --dataset uncertain-prostate \
    --loss-type dice \
    --nchannels 1 \
    --model multi-unet \
    --nclass 3 \
    --task-num 0

python train.py \
    --workers 4 \
    --lr 0.01 \
    --epochs 200 \
    --gpu-ids 0,1,2,3 \
    --batch-size 4 \
    --test-batch-size 1 \
    --checkname multi-unet-prostate0 \
    --eval-interval 1 \
    --dataset uncertain-prostate \
    --loss-type dice \
    --nchannels 1 \
    --model decoder-unet \
    --nclass 3 \
    --task-num 0

python train.py \
    --workers 4 \
    --lr 0.01 \
    --epochs 200 \
    --gpu-ids 0,1,2,3 \
    --batch-size 4 \
    --test-batch-size 1 \
    --checkname multi-unet-prostate0 \
    --eval-interval 1 \
    --dataset uncertain-prostate \
    --loss-type dice \
    --nchannels 1 \
    --model attn-unet \
    --nclass 3 \
    --task-num 0
