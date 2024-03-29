python train_bayesian.py \
       --workers 4 \
        --lr 0.001 \
       --epochs 2000 \
       --gpu-ids 0 \
       --batch-size 32 \
       --test-batch-size 1 \
       --checkname voemd_unet_lidc \
       --eval-interval 4 \
       --dataset lidc \
       --loss-type vELBO \
       --nchannels 1 \
       --model voemd-unet \
       --nclass 1 \
       --task-num 0 \
       --beta-type 0.1 \
       --save-path /data/ssd/qingqiao/BOEMD_run_test
