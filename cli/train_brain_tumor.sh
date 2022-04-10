python train.py \
    mode=train \
    epochs=84 \
    model=boemd \
    model.num_samples=16 \
    model.attention="attn" \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=elbo \
    loss.beta_type="Standard" \
    batch_size=16 \
    test_batch_size=1 \
    dataset=qubiq \
    dataset.task="brain-tumor" \
    dataset.task_id=0 \
    checkname=boemd \
    save_path=/data/ssd/${USER}/BOEMD_run_test \
    optim=adam \
    seed=2021 \
    optim.lr=1e-3 \

python train.py \
    mode=train \
    epochs=80 \
    model=unet\
    model.dropout=True \
    model.drop_p=0.3 \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=16 \
    test_batch_size=4 \
    dataset=qubiq \
    dataset.task="brain-tumor" \
    dataset.task_id=0 \
    checkname=unet\
    save_path=/data/ssd/${USER}/BOEMD_run_test \
    optim=adam \
    seed=42 \
    optim.lr=1e-3 \


python train.py \
    mode=train \
    epochs=80 \
    model=multi_unet\
    model.dropout=True \
    model.drop_p=0.3 \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=16 \
    test_batch_size=4 \
    dataset=qubiq \
    dataset.task="brain-tumor" \
    dataset.task_id=0 \
    checkname=multi_unet\
    save_path=/data/ssd/${USER}/BOEMD_run_test \
    optim=adam \
    seed=42 \
    optim.lr=1e-3 \

python train.py \
    mode=train \
    epochs=80 \
    model=oemd\
    model.dropout=True \
    model.drop_p=0.3 \
    model.attention=NULL \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=16 \
    test_batch_size=4 \
    dataset=qubiq \
    dataset.task="brain-tumor" \
    dataset.task_id=0 \
    checkname=decoder_unet\
    save_path=/data/ssd/${USER}/BOEMD_run_test \
    optim=adam \
    seed=42 \
    optim.lr=1e-3 \


python train.py \
    mode=train \
    epochs=80 \
    model=oemd\
    model.dropout=True \
    model.drop_p=0.3 \
    model.attention='attn' \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=16 \
    test_batch_size=4 \
    dataset=qubiq \
    dataset.task="brain-tumor" \
    dataset.task_id=0 \
    checkname=decoder_attn_unet\
    save_path=/data/ssd/${USER}/BOEMD_run_test \
    optim=adam \
    seed=42 \
    optim.lr=1e-3 \
