python train.py \
    mode=train \
    epochs=200 \
    model=boemd \
    model.num_samples=16 \
    model.attention="attn" \
    eval_interval=1 \
    gpu_ids="'0,1'" \
    loss=elbo \
    loss.beta_type=0.001 \
    batch_size=32 \
    test_batch_size=16 \
    dataset=lidc-patient \
    checkname=boemd_lidc_patient_double_check \
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=adam \
    seed=2021 \
    optim.lr=5e-3 \
