python train.py \
    -m \
    mode=test \
    epochs=300 \
    model=phiseg \
    model.num_samples=4 \
    gpu_ids="'0,1'" \
    batch_size=64 \
    test_batch_size=16 \
    dataset=lidc-patient \
    dataset.mode="ged" \
    checkname=phiseg-lidc-patient \
    save_path=/data/ssd/${USER}/bOEMD_results/ \
    apply_sigmoid="False" \
    optim=adam \
    optim.lr=1e-3 \
    optim.weight_decay=1e-5 \
    resume="/data/ssd/wanghao/bOEMD_results/lidc-patient/phiseg-lidc-patient/experiment_00/checkpoint.pth.tar"
