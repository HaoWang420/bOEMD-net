python train.py \
    mode=train \
    epochs=200 \
    model=boemd \
    gpu_ids="'0'" \
    loss=elbo \
    elbo.beta_type=
    batch_size=32 \
    test_batch_size=1 \
    dataset=lidc-small \
    checkname=boemd_lidc_small \
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=adam \
    optim.lr=1e-3
