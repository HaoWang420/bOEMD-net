python train_bayesian.py \
       --workers 4 \
       --lr 0.001 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 4 \
       --test-batch-size 1 \
       --checkname multi-atten-bunet \
       --eval-interval 4 \
       --dataset uncertain-brain-growth \
       --loss-type ELBO \
       --nchannels 1 \
       --model multi-atten-bunet \
       --nclass 3 \
       --task-num 0

python train_bayesian.py \
       --workers 4 \
       --lr 0.001 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 4 \
       --test-batch-size 1 \
       --checkname multi-atten-bunet \
       --eval-interval 4 \
       --dataset uncertain-brats \
       --loss-type ELBO \
       --nchannels 4 \
       --model multi-atten-bunet \
       --nclass 3 \
       --task-num 0

python train_bayesian.py \
       --workers 4 \
       --lr 0.001 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 4 \
       --test-batch-size 1 \
       --checkname multi-atten-bunet \
       --eval-interval 4 \
       --dataset uncertain-brats \
       --loss-type ELBO \
       --nchannels 4 \
       --model multi-atten-bunet \
       --nclass 3 \
       --task-num 1

python train_bayesian.py \
       --workers 4 \
       --lr 0.001 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 4 \
       --test-batch-size 1 \
       --checkname multi-atten-bunet \
       --eval-interval 4 \
       --dataset uncertain-brats \
       --loss-type ELBO \
       --nchannels 4 \
       --model multi-atten-bunet \
       --nclass 3 \
       --task-num 2

python train_bayesian.py \
       --workers 4 \
       --lr 0.001 \
       --epochs 200 \
       --gpu-ids 0 \
       --batch-size 4 \
       --test-batch-size 1 \
       --checkname multi-atten-bunet \
       --eval-interval 4 \
       --dataset uncertain-kidney \
       --loss-type ELBO \
       --nchannels 1 \
       --model multi-atten-bunet \
       --nclass 3 \
       --task-num 0

 python train_bayesian.py \
         --workers 2 \
         --lr 0.001 \
         --epochs 200 \
         --gpu-ids 0 \
         --batch-size 1 \
         --test-batch-size 1 \
         --checkname multi-atten-bunet \
         --eval-interval 4 \
         --dataset uncertain-prostate \
         --loss-type ELBO \
         --nchannels 1 \
         --model multi-atten-bunet \
         --nclass 3 \
         --task-num 0

 python train_bayesian.py \
         --workers 2 \
         --lr 0.001 \
         --epochs 200 \
         --gpu-ids 0 \
         --batch-size 1 \
         --test-batch-size 1 \
         --checkname multi-atten-bunet \
         --eval-interval 4 \
         --dataset uncertain-prostate \
         --loss-type ELBO \
         --nchannels 1 \
         --model multi-atten-bunet \
         --nclass 3 \
         --task-num 1
