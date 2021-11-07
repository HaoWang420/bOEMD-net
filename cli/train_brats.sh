python train.py \
    -m \
    mode=train \
    epochs=200 \
    model=oemd \
    model.dropout=True \
    model.drop_p=0.5 \
    loss=dice \
    batch_size=4 \
    test_batch_size=1 \
    dataset=qubiq \
    dataset.task=brain-tumor \
    dataset.task_id=0 \
    checkname=brats \
    save_path=/data/ssd/wanghao/bOEMD_results/ \
    optim=sgd \
    optim.lr=1e-1
