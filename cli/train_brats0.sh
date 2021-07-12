for ii in {1..4}
do
    python train.py \
        --workers 4 \
        --lr 0.01 \
        --epochs 200 \
        --gpu-ids 0,1 \
        --batch-size 4 \
        --test-batch-size 1 \
        --checkname multi-unet-brats0_${ii} \
        --eval-interval 1 \
        --dataset uncertain-brats \
        --loss-type dice \
        --nchannels 4 \
        --model unet \
        --nclass 3 \
        --task-num 0

    python train.py \
        --workers 4 \
        --lr 0.01 \
        --epochs 200 \
        --gpu-ids 0,1 \
        --batch-size 4 \
        --test-batch-size 1 \
        --checkname multi-unet-brats0_${ii} \
        --eval-interval 1 \
        --dataset uncertain-brats \
        --loss-type dice \
        --nchannels 4 \
        --model multi-unet \
        --nclass 3 \
        --task-num 0

    python train.py \
        --workers 4 \
        --lr 0.01 \
        --epochs 200 \
        --gpu-ids 0,1 \
        --batch-size 4 \
        --test-batch-size 1 \
        --checkname multi-unet-brats0_${ii} \
        --eval-interval 1 \
        --dataset uncertain-brats \
        --loss-type dice \
        --nchannels 4 \
        --model decoder-unet \
        --nclass 3 \
        --task-num 0

    python train.py \
        --workers 4 \
        --lr 0.01 \
        --epochs 200 \
        --gpu-ids 0,1 \
        --batch-size 4 \
        --test-batch-size 1 \
        --checkname multi-unet-brats0_${ii} \
        --eval-interval 1 \
        --dataset uncertain-brats \
        --loss-type dice \
        --nchannels 4 \
        --model attn-unet \
        --nclass 3 \
        --task-num 0

    python train.py \
        --workers 4 \
        --lr 0.01 \
        --epochs 200 \
        --gpu-ids 0,1 \
        --batch-size 4 \
        --test-batch-size 1 \
        --checkname multi-unet-brats0_${ii} \
        --eval-interval 1 \
        --dataset uncertain-brats \
        --loss-type dice \
        --nchannels 4 \
        --model pattn-unet-al \
        --nclass 3 \
        --task-num 0
done
