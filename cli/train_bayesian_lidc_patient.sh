python train.py \
    mode=train \
    epochs=250 \
    model=boemd \
    model.num_sample=50 \
    model.attention="attn" \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=elbo \
    loss.beta_type=0.001 \
    batch_size=32 \
    test_batch_size=1 \
    dataset=lidc-patient \
    checkname=boemd_lidc_patient \
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=adam \
    seed=42 \
    optim.lr=1e-3 \
