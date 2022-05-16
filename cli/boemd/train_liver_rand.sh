python train.py \
    -m \
    mode=train \
    epochs=500 \
    model=boemd \
    model.num_samples=16 \
    gpu_ids="'0,1'" \
    batch_size=8 \
    eval_interval=5 \
    test_batch_size=1 \
    dataset=liver-rand \
    loss='elbo' \
    checkname=boemd_liver \
    optim=adam \
    optim.lr=1e-3 \
    optim.weight_decay=1e-5
