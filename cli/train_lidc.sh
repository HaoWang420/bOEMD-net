python train.py \
    -m \
    mode=train \
    epochs=300 \
    model=boemd \
    gpu_ids="'0,1'" \
    loss=elbo \
    loss.weights.kl=0.001 \
    batch_size=64 \
    test_batch_size=64 \
    dataset=lidc \
    dataset.train_ratio=0.1 \
    checkname=boemd-0.1-train \
    save_path=/data/ssd/wanghao/bOEMD_results/ \
    optim=adam \
    optim.lr=1e-2
