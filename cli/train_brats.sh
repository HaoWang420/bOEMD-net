python train.py \
    -m \
    mode=train \
    epochs=200 \
    model=unet \
    loss=dice \
    batch_size=4 \
    test_batch_size=4 \
    optim=adam \
    optim.lr=1e-2 \
    dataset=qubiq \
    dataset.task=brain-tumor \
    dataset.task_id=0 \
    checkname=test \
    optim.lr=1e-2
