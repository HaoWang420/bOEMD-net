python train.py \
    -m \
    mode=train \
    epochs=150 \
    model=boemd \
    model.num_samples=25 \
    gpu_ids="'0'" \
    batch_size=4 \
    eval_interval=5 \
    test_batch_size=1 \
    dataset=liver \
    dataset.mode='fix' \
    loss='elbo' \
    checkname=boemd_liver_3_rater \
    save_path=/data/sdb/${USER}/BOEMD_run_test/ \
    optim=adam \
    optim.lr=1e-3 \
    optim.weight_decay=1.e-5 \
    # resume="/data/sdb/qingqiao/BOEMD_run_test/liver/boemd_liver_3_rater/experiment_04/checkpoint.pth.tar"
