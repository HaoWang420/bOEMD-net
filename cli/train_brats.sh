python train.py \
    -m \
    mode=test \
    epochs=200 \
    model=oemd \
    gpu_ids="'0,1'" \
    model.dropout=True \
    model.drop_p=0.2,0.3 \
    loss=dice \
    batch_size=64 \
    test_batch_size=64 \
    dataset=lidc \
    checkname=oemd-dropout \
    save_path=/data/ssd/wanghao/bOEMD_results/ \
    optim=sgd \
    optim.lr=1e-2
