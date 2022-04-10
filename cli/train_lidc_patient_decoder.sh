python train.py \
    mode=train \
    epochs=200 \
    model=unet\
    model.dropout=True \
    model.drop_p=0.3 \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=32 \
    test_batch_size=32 \
    dataset=lidc-patient \
    checkname=lidc_patient_unet_dropout\
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=sgd \
    seed=42 \
    optim.lr=1e-3 \
    resume="/data/ssd/qingqiao/BOEMD_run_test/lidc-patient/lidc_patient_unet/experiment_00/checkpoint.pth.tar"

python train.py \
    mode=test \
    epochs=200 \
    model=multi_unet\
    model.dropout=False \
    model.drop_p=0.3 \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=32 \
    test_batch_size=32 \
    dataset=lidc-patient \
    checkname=lidc_patient_multi_unet_dropout\
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=sgd \
    seed=42 \
    optim.lr=1e-3 \
    resume="/data/ssd/qingqiao/BOEMD_run_test/lidc-patient/lidc_patient_multi_unet/experiment_00/checkpoint.pth.tar"

python train.py \
    mode=test \
    epochs=200 \
    model=oemd\
    model.dropout=True \
    model.drop_p=0.3 \
    model.attention=NULL \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=32 \
    test_batch_size=32 \
    dataset=lidc-patient \
    checkname=lidc_patient_decoder_unet_dropout\
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=sgd \
    seed=42 \
    optim.lr=1e-3 \
    resume="/data/ssd/qingqiao/BOEMD_run_test/lidc-patient/lidc_patient_decoder_unet/experiment_01/checkpoint.pth.tar"

python train.py \
    mode=test \
    epochs=200 \
    model=oemd\
    model.dropout=False \
    model.drop_p=0.3 \
    model.attention='attn' \
    eval_interval=5 \
    gpu_ids="'0'" \
    loss=dice \
    batch_size=8 \
    test_batch_size=8 \
    dataset=lidc-patient \
    checkname=lidc_patient_decoder_attn_unet_dropout\
    save_path=/data/ssd/qingqiao/BOEMD_run_test \
    optim=sgd \
    seed=42 \
    optim.lr=1e-3 \
    resume="/data/ssd/qingqiao/BOEMD_run_test/lidc-patient/lidc_patient_decoder_attn_unet/experiment_00/checkpioont.pth.tar"