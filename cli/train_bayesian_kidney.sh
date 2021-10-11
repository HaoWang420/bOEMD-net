python train_bayesian.py \
       --workers 4 \
       --lr 0.01 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 4 \
       --test-batch-size 1 \
       --checkname batten_unet_kidney \
       --eval-interval 5 \
       --dataset uncertain-kidney \
       --loss-type ELBO \
       --nchannels 1 \
       --model batten-unet \
       --nclass 1 \
       --task-num 0

