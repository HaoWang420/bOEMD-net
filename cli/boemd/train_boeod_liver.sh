python train.py \
    -m \
    mode=train \
    epochs=410 \
    model=boeod \
    model.num_samples=16 \
    gpu_ids="'0'" \
    batch_size=4 \
    eval_interval=5 \
    test_batch_size=1 \
    dataset=liver \
    loss='elbo' \
    checkname=boeod_liver \
    optim=adam \
    optim.lr=1e-3 \
    optim.weight_decay=1e-5
