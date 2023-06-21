python train.py \
    -m \
    mode=test \
    epochs=150 \
    model=boeod \
    model.num_samples=3 \
    gpu_ids="'0'" \
    batch_size=4 \
    eval_interval=5 \
    test_batch_size=1 \
    dataset=liver \
    dataset.mode='choice' \
    loss='elbo' \
    checkname=boeod_liver_3_rater \
    save_path=/data/sdb/${USER}/BOEMD_run_test/ \
    optim=adam \
    optim.lr=1e-4 \
    optim.weight_decay=1e-5 \
    resume="/data/sdb/qingqiao/BOEMD_run_test/liver/boeod_liver_3_rater/experiment_02/checkpoint.pth.tar"
