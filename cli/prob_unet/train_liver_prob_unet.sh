python train.py \
    -m \
    mode=train \
    epochs=300 \
    model=prob-unet \
    model.num_samples=3 \
    model.img_size='[1,512,512]' \
    gpu_ids="'0'" \
    batch_size=4 \
    test_batch_size=1 \
    dataset=liver \
    dataset.mode='choice_p' \
    checkname=prob-unet-liver-tumor \
    save_path=/data/sdb/${USER}/BOEMD_run_test/ \
    apply_sigmoid="False" \
    optim=adam \
    optim.lr=1e-4 \
    optim.weight_decay=1e-5
