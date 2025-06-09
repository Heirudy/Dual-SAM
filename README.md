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

## 2. Data Preparation

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
For 3D datasets
```
python tranin.py 
```
For 2D datasets
```
python tranin_2D.py 
```

##  Supplementary Materials


### Additional experimental details
#### 1. Sensitivity analysis of loss weights λ₁ and λ₂

![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image1.png)


#### 2. Comparative Visualizations of LEPF (Fourier Transform) vs. Wavelet Transform

![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image2.png)


#### 3. Step-by-step pseudocode for Cross-Path Prompt Fusion (CPF)

![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image3.png)


#### 4. Visual comparisons of predicted masks from two decoders at multiple training stages

![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image4.png)


#### 5. Sensitivity analysis of different LoRA ranks on Chest X-ray segmentation performance

![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image5.png)


#### 6. Comparison of cardiac structure segmentation and clinical diagnostic indicators calculation on ACDC dataset. 

![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image6.png)



### Codebase and all experimental logs
<span style="color:red">The comparative experiment codebase and its log files are available in the "Comparison Method" folder.</span>  


