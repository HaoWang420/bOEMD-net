python train.py \
    -m \
    mode=train \
    epochs=300 \
    model=prob-unet \
    model.num_samples=6 \
    model.img_size='[1,640,640]' \
    gpu_ids="'0'" \
    batch_size=4 \
    test_batch_size=1 \
    dataset=qubiq \
    dataset.task='prostate' \
    dataset.task_id=1 \
    dataset.mode="ged" \
    checkname=prob-unet-prostate-task-1 \
    apply_sigmoid="False" \
    optim=adam \
    optim.lr=1e-4 \
    optim.weight_decay=1e-5
