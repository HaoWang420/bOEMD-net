python train.py \
    -m \
    mode=train \
    epochs=250 \
    model=oemd\
    gpu_ids="'0,1'" \
    model.dropout=False,False \
    model.drop_p=0.2,0.2 \
    model.attention=None,'attn'\
    loss=dice \
    batch_size=32 \
    test_batch_size=32 \
    dataset=lidc-small \
    checkname=lidc-small-oemd,lidc-small-attn-oemd \
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=sgd \
    seed=1 \
    optim.lr=0.5
