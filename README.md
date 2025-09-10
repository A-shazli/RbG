# Registration by Generation
![Framwork](https://github.com/user-attachments/assets/3459a3e6-9db6-4a2c-9469-a1aef20e0669)


## 1. Description

- Templates from https://github.com/ashleve/lightning-hydra-template was used.

- `Pytorch-lighting` + `Hydra` + `Tensorboard` was used for experiment-tracking


## 2. Dataset
#### Dataset 
Grand challenge ['SynthRAD 2023'](https://synthrad2023.grand-challenge.org/) Pelvis MR, CT

#### Preprocessing
- CT & CBCT: 
  - 5%, 95% percentile clip 
  - z-score norm whole patient 
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
