python train.py \
    -m \
    mode=train \
    epochs=300 \
    model="prob_unet" \
    model.num_samples=4 \
    gpu_ids="'0,1'" \
    batch_size=64 \
    test_batch_size=16 \
    dataset=lidc-patient \
    dataset.mode="ged" \
    checkname=prob-unet-lidc-patient \
    save_path=/data/ssd/${USER}/bOEMD_results/ \
    apply_sigmoid="False" \
    optim=adam \
    optim.lr=1e-3 \
    optim.weight_decay=1e-5
