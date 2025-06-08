# Dual-SAM: Prompt-enhanced Dual-branch SAM for Robust Semi-Supervised Medical ImageSegmentation


## Overall Framework ##
![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image7.png)


## Setup
```bash
git clone https://github.com/Heirudy/Dual-SAM.git
cd ESP-MedSAM
```

## 📆 TODO

- [x] Release code

## 🎮 Getting Started

### 1. Install Environment

see [INSTALL](INSTALL.md).

### 2. Prepare Dataset and Checkpoints

see [PREPARE](PREPARE.md).

### 3. Adapt with Weak Supervision

```
# 1 modify configs/config.py 
# Prompt type: box, point, coarse

# 2 adapt
python adaptation.py
```

### 4. Validation

```
python validate.py --ckpt /path/to/checkpoint
```


## 🖼️ Visualization

<div align="center">
<img width="800" alt="image" src="asserts/VISUAL.webp?raw=true">
</div>



## 💡 Acknowledgement

- [SAM](https://github.com/facebookresearch/segment-anything)

- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)

- [SAM-LoRA](https://github.com/JamesQFreeman/Sam_LoRA)


