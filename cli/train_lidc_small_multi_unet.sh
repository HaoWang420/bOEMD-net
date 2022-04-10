python train.py \
    mode=train \
    epochs=250 \
    model=multi_unet\
    gpu_ids="'0'" \
    model.dropout=False \
    model.drop_p=0.2 \
    model.attention=None\
    loss=dice \
    batch_size=32 \
    test_batch_size=32 \
    dataset=lidc-small \
    checkname=lidc-small-multi-unet \
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=sgd \
    seed=1 \
    optim.lr=1e-2
