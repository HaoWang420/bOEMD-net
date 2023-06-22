python train.py \
    -m \
    mode=test \
    epochs=150 \
    model=boemd \
    model.num_samples=1 \
    gpu_ids="'0'" \
    batch_size=4 \
    eval_interval=5 \
    test_batch_size=1 \
    dataset=liver \
    dataset.mode="random" \
    loss='elbo' \
    checkname=boemd_liver_3_rater_permutation_e_-5 \
    save_path=/data/sdb/${USER}/BOEMD_run_test/ \
    optim=adam \
    optim.lr=1e-5 \
    optim.weight_decay=1e-5 \
    resume="/data/sdb/qingqiao/BOEMD_run_test/liver/boemd_liver_3_rater_permutation_e_-5/experiment_03/checkpoint.pth.tar"
