python train_bayesian.py \
       --workers 4 \
       --lr 0.01 \
       --epochs 100 \
       --gpu-ids 0 \
       --batch-size 32 \
       --test-batch-size 1 \
       --checkname lidc-syn-shuffle \
       --eval-interval 5 \
       --dataset lidc-syn \
       --loss-type ELBO \
       --nchannels 1 \
       --model batten-unet \
       --nclass 1 \
       --task-num 0 \
       --save-path /data/ssd/wanghao/bOEMD_run 
