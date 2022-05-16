python train.py \
    mode=test \
    epochs=200 \
    model=boemd \
    model.num_samples=16 \
    model.attention="attn" \
    eval_interval=5 \
    gpu_ids="'0,1'" \
    loss=elbo \
    loss.beta_type=0.00000001 \
    batch_size=32 \
    test_batch_size=1 \
    dataset=lidc-patient \
    checkname=boemd_lidc_patient_lr_5e-3_e-8beta \
    save_path=/data/ssd/${USER}/BOEMD_run_test \
    optim=adam \
    seed=2021 \
    optim.lr=5e-3 \

