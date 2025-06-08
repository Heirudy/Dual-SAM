# Dual-SAM: Prompt-enhanced Dual-branch SAM for Robust Semi-Supervised Medical ImageSegmentation
##Overall Framework ##
![image](https://github.com/Heirudy/Dual-SAM/blob/main/image/image7.png)


## 🛠Setup

```bash
git clone https://github.com/Heirudy/Dual-SAM.git
cd ESP-MedSAM

**Note**: Please refer to requirements.txt


## 📚Data Preparation

The structure is as follows.
```
ESP-MedSAM
├── datasets
│   ├── image_1024
│     ├── ISIC_0000000.png
|     ├── ...
|   ├── mask_1024
│     ├── ISIC_0000000.png
|     ├── ...
```

## 🎪Segmentation Model Zoo
We provide all pre-trained models here.
| MA-Backbone | MC | Checkpoints |
|-----|------|-----|
|TinyViT| Dermoscopy | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=dLxPsT)|
|TinyViT| X-ray | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=dLxPsT)|
|TinyViT| Fundus | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=dLxPsT)|
|TinyViT| Colonoscopy | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=dLxPsT)|
|TinyViT| Ultrasound | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=dLxPsT)|
|TinyViT| Microscopy | [Link](https://nottinghamedu1-my.sharepoint.com/:f:/g/personal/scxqx1_nottingham_edu_cn/EpOiC2oi8eFFuUqCK_JlSD0BKCn0oHYsVW83sg-fzZUx6w?e=dLxPsT)|

## 📜Citation
If you find this work helpful for your project, please consider citing the following paper:
```
@article{xu2024esp,
  title={ESP-MedSAM: Efficient Self-Prompting SAM for Universal Domain-Generalized Medical Image Segmentation},
  author={Xu, Qing and Li, Jiaxuan and He, Xiangjian and Liu, Ziyu and Chen, Zhen and Duan, Wenting and Li, Chenxin and He, Maggie M and Tesema, Fiseha B and Cheah, Wooi P and others},
  journal={arXiv preprint arXiv:2407.14153},
  year={2024}
}
```
