python train.py \
    -m \
    mode=train \
    epochs=300 \
    model=phiseg \
    model.num_samples=3 \
    model.img_size='[1,512,512]' \
    gpu_ids="'0'" \
    batch_size=4 \
    test_batch_size=1 \
    dataset=qubiq \
    dataset.task='kidney' \
    dataset.task_id=0 \
    dataset.mode="ged" \
    checkname=phiseg-kidney-task-0 \
    save_path=/data/ssd/${USER}/bOEMD_results/ \
    apply_sigmoid="False" \
    optim=adam \
    optim.lr=1e-3 \
    optim.weight_decay=1e-5
