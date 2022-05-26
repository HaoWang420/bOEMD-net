python train.py \
    -m \
    mode=train \
    epochs=500 \
    model=prob-unet \
    model.num_samples=7 \
    model.img_size='[1,256,256]' \
    gpu_ids="'0'" \
    batch_size=4 \
    test_batch_size=1 \
    dataset=qubiq \
    dataset.task='brain-growth' \
    dataset.task_id=0 \
    dataset.mode="ged" \
    checkname=prob-unet-brain-growth-task-0 \
    apply_sigmoid="False" \
    optim=adam \
    optim.lr=1e-4 \
    optim.weight_decay=1e-5
