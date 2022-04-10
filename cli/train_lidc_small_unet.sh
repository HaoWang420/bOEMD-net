python train.py \
    mode=train \
    epochs=250 \
    model=unet\
    gpu_ids="'0'" \
    model.dropout=False \
    model.drop_p=0 \
    loss=dice \
    batch_size=32 \
    test_batch_size=32 \
    dataset=lidc-small \
    checkname=lidc-small-unet \
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=sgd \
    seed=1 \
    optim.lr=0.5
