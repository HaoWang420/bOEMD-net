python train.py \
    -m \
    mode=train \
    epochs=500 \
    model=boemd \
    model.num_samples=1 \
    gpu_ids="'0,1'" \
    batch_size=4 \
    test_batch_size=1 \
    dataset=qubiq \
    dataset.task='prostate' \
    dataset.task_id=0 \
    loss='elbo'
    checkname=boemd-prostate-task-0 \
    save_path=/data/ssd/${USER}/bOEMD_results/ \
    optim=adam \
    optim.lr=1e-3 \
    optim.weight_decay=1e-5

