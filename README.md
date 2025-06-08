# Dual-SAM: Prompt-enhanced Dual-branch SAM for Robust Semi-Supervised Medical ImageSegmentation


## Overall Framework ##
![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image7.png)


## Setup
```bash
git clone https://github.com/Heirudy/Dual-SAM.git
```

##  Getting Started

### 1. Install Environment

see requirements.txt

## Data Preparation

The structure is as follows.
```
Dual-SAM
├── datasets
│   ├── image
│     ├── image0.png
|     ├── ...
|   ├── mask
│     ├── image0.png
|     ├── ...
```

### 3. train

```
python tranin.py 
```

```
python tranin_2D.py 
```


##  Supplementary experiments

- [SAM](https://github.com/facebookresearch/segment-anything)

- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)

- [SAM-LoRA](https://github.com/JamesQFreeman/Sam_LoRA)


