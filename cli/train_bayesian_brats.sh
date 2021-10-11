python train_bayesian.py \
       --workers 4 \
       --lr 0.01 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 4 \
       --test-batch-size 1 \
       --checkname batten_unet_brats1 \
       --eval-interval 5 \
       --dataset uncertain-brats \
       --loss-type ELBO \
       --nchannels 4 \
       --model batten-unet \
       --nclass 1 \
       --task-num 0

# CUDA_VISIBLE_DEVICES=0 \
# python train_bayesian.py \
#        --workers 4 \
#        --lr 0.001 \
#        --epochs 200 \
#        --gpu-ids 0 \
#        --batch-size 4 \
#        --test-batch-size 1 \
#        --checkname batten_unet_brats2 \
#        --eval-interval 4 \
#        --dataset uncertain-brats \
#        --loss-type ELBO \
#        --nchannels 4 \
#        --model batten-unet \
#        --nclass 3 \
#        --task-num 2
# 
# CUDA_VISIBLE_DEVICES=0 \
# python train_bayesian.py \
#        --workers 4 \
#        --lr 0.001 \
#        --epochs 200 \
#        --gpu-ids 0 \
#        --batch-size 4 \
#        --test-batch-size 1 \
#        --checkname batten_unet \
#        --eval-interval 4 \
#        --dataset uncertain-brain-growth \
#        --loss-type ELBO \
#        --nchannels 1 \
#        --model batten-unet \
#        --nclass 3 \
#        --task-num 0
# 
# CUDA_VISIBLE_DEVICES=0 \
# python train_bayesian.py \
#        --workers 4 \
#        --lr 0.001 \
#        --epochs 200 \
#        --gpu-ids 0 \
#        --batch-size 4 \
#        --test-batch-size 1 \
#        --checkname batten_unet \
#        --eval-interval 4 \
#        --dataset uncertain-kidney \
#        --loss-type ELBO \
#        --nchannels 1 \
#        --model batten-unet \
#        --nclass 3 \
#        --task-num 0

# CUDA_VISIBLE_DEVICES=0 \
#  python train_bayesian.py \
#          --workers 2 \
#          --lr 0.001 \
#          --epochs 200 \
#          --gpu-ids 0 \
#          --batch-size 1 \
#          --test-batch-size 1 \
#          --checkname uncertain-prostate0 \
#          --eval-interval 4 \
#          --dataset uncertain-prostate \
#          --loss-type ELBO \
#          --nchannels 1 \
#          --model batten-unet \
#          --nclass 3 \
#          --task-num 0
# 
# CUDA_VISIBLE_DEVICES=0 \
#  python train_bayesian.py \
#          --workers 2 \
#          --lr 0.001 \
#          --epochs 200 \
#          --gpu-ids 0 \
#          --batch-size 1 \
#          --test-batch-size 1 \
#          --checkname uncertain-prostate1 \
#          --eval-interval 4 \
#          --dataset uncertain-prostate \
#          --loss-type ELBO \
#          --nchannels 1 \
#          --model batten-unet \
#          --nclass 3 \
#          --task-num 1
