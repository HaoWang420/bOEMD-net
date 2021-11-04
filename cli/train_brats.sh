python train.py \
    -m \
    mode=train \
    epochs=200 \
    model=oemd \
    model.dropout=True \
    model.drop_p=0.1,0.2,0.5 \
    loss=dice \
    batch_size=64 \
    test_batch_size=64 \
    optim=adam \
    optim.lr=1e-2 \
    dataset=lidc \
    checkname=test \
    save_path=/data/ssd/wanghao/bOEMD_results/ \
    optim.lr=1e-2
