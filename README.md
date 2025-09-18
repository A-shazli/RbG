# Registration by Generation
<img width="5171" height="3797" alt="Graphical abstract" src="https://github.com/user-attachments/assets/b7ac5e47-444b-4649-8b69-9e62423bbc44" />



## 1. Description

- Templates from https://github.com/ashleve/lightning-hydra-template was used.

- `Pytorch-lighting` + `Hydra` + `Tensorboard` was used for experiment-tracking


## 2. Dataset
#### Dataset 
Grand challenge ['SynthRAD 2023'](https://synthrad2023.grand-challenge.org/) Brain CBCT, CT

#### Preprocessing
- CT & CBCT: 
  - Non-local means denoising 
  - Histogram Equalization 
  - -1 ~ 1 minmax norm whole patient

Two sets are provided in the data folder, one for registering CT to CBCT images and one for registering CT to SynCT (generated from CBCT) iamges.

#### File Format: 
h5 formate is used to store and load the data.


## 4. How to run

```bash
#### Training
python train.py tags='SynthRAD_RbG'
```
Tags can be edited accordingly to switch for example between the different attention types. (MEA (Memory efficient attention) or Softmax)
