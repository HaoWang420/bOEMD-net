# bOEMD-Net

The *project* pape is [here](https://winstonhutiger.github.io/project/qubiq_uncertainty_quanification/). 

This is the official code repository for [*Inter-Rater Uncertainty Quantification in Medical Image Segmentation via Rater-Specific Bayesian Neural Networks*](https://arxiv.org/pdf/2306.16556v1.pdf).

In this work, we present a simple yet effective Bayesian neural network architecture to estimate the inter-rater uncertainty in medical image segmentation.


## Usage

### 1. Dataset Preparation

#### Liver dataset 
Please download the dataset from [here](https://drive.google.com/file/d/1DVzHYt5OM9eWaMu0eC1no31plEXEGCkQ/view?usp=sharing)

This dataset is annotated by Zhiheng Zhang, Jan Kirschke and Benedikt Wiestler. 

**Huge shout for their contribution to this work!!!**

#### Qubiq challenge dataset
The train and validation data could be obtained [here](https://qubiq.grand-challenge.org/)

#### LIDC-IDRI dataset
The preprocessed LIDC-IDRI dataset can be downloaded from [the bucket from Deepmind](https://console.cloud.google.com/storage/browser/hpunet-data/lidc_crops/).



### 2. Training & Evalutation
To reproduce our results on LIDC-IDRI dataset, please run:
```
bash cli/boemd/train_bomd_lidc_patient.sh
```

## License
Our work is released under the MIT license. Please check the [LICENSE](LICENSE) for more information.

## Citation
If you find our work is helpful for your research, please cite our code repository in your work. 

## Acknowledgement
Phiseg code is based on [Phiseg-code](https://github.com/baumgach/PHiSeg-code) and built upon [UNet-Zoo](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/).

Probabilistic Unet code is based on [probabilistic_unet](https://github.com/SimonKohl/probabilistic_unet), [Probabilistic-Unet-Pytorch](https://github.com/stefanknegt/Probabilistic-Unet-Pytorh) and [UNet-Zoo](https://github.com/gigantenbein/UNet-Zoo).

Bayesian CNN is based on [PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN).

Bayes by Backprop is based on [Bayesian-Neural-Networks](https://github.com/JavierAntoran/Bayesian-Neural-Networks).

Preprocessed patient-id specific LICD-IDRI dataset is from [hierarchical_probabilistic_unet](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_probabilistic_unet).
