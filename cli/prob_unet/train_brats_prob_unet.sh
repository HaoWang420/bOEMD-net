python train.py \
    -m \
    mode=train \
    epochs=500 \
    model=prob-unet \
    model.num_samples=3 \
    model.img_size='[4,256,256]' \
    gpu_ids="'0'" \
    batch_size=4 \
    test_batch_size=1 \
    dataset=qubiq \
    dataset.task='brain-tumor' \
    dataset.task_id=1 \
    dataset.mode="ged" \
    checkname=prob-unet-brain-tumor-task-1 \
    apply_sigmoid="False" \
    optim=adam \
    optim.lr=1e-4 \
    seed=42
    # optim.weight_decay=1e-5
