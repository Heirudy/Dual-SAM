
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic
from scipy.stats import entropy


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt, raw_spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt, raw_spacing)
        hd95 = metric.binary.hd95(pred, gt, raw_spacing)
    else:
        print('bad')
        asd = -1
        hd95 = -1
    jc = metric.binary.jc(pred, gt)
    return dice, hd95, asd, jc

def calculate_metric_percase_nospacing(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    else:
        print('bad')
        asd = -1
        hd95 = -1
    jc = metric.binary.jc(pred, gt)
    return dice, hd95, asd, jc

def calculate_metric_percase_nan(pred, gt, raw_spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt, raw_spacing)
        hd95 = metric.binary.hd95(pred, gt, raw_spacing)
    else:
        asd = np.nan
        hd95 = np.nan
    jc = metric.binary.jc(pred, gt)
    return dice, hd95, asd, jc


def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0, patch_size[0], patch_size[1]
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            net.eval()
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks = outputs['masks']
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred

        # get resolution
        case_raw = 'xxx/data/ACDC_raw/' + case+ '.nii.gz'
        case_raw = sitk.ReadImage(case_raw)
        raw_spacing = case_raw.GetSpacing()
        raw_spacing_new = []
        raw_spacing_new.append(raw_spacing[2])
        raw_spacing_new.append(raw_spacing[1])
        raw_spacing_new.append(raw_spacing[0])
        raw_spacing = raw_spacing_new

    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i,raw_spacing))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        #sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        #sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        print('saved successfully!')
    return metric_list

def test_single_volume_mean(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0, patch_size[0], patch_size[1]
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            # custom_mask=AdaptiveMaskNet()

            net.eval()
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks1 = outputs['masks']
                output_masks2 = outputs['masks2']
                output_masks1 = torch.softmax(output_masks1, dim=1)
                output_masks2 = torch.softmax(output_masks2, dim=1)
                output_masks=custom_mask( output_masks1,  output_masks2)

                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
            # 从 case 中提取患者编号和帧编号
            patient_id, frame_id = case.split("_")  # 假设 case 格式为 "patient003_frame01"

            # 构造预测结果的文件名
            pred_name = f"{patient_id}_{frame_id}_pred_{ind + 1}.png"
            pred_path = os.path.join(test_save_path, pred_name)

            # 保存预测结果
            plt.figure(figsize=(10, 10))
            plt.imshow(pred, cmap='gray', interpolation='none')
            plt.axis('off')
            plt.savefig(pred_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
            plt.close('all')

            # 保存 pred 和 label 的对比图
            _, axs = plt.subplots(1, 2, figsize=(25, 25))
            axs[0].imshow(pred, cmap='gray', interpolation='none')
            axs[0].set_title('Prediction')
            axs[0].axis('off')

            axs[1].imshow(label[ind, :, :], cmap='gray', interpolation='none')
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')

            plt.subplots_adjust(wspace=0.01, hspace=0)
            filename = f"{patient_id}_{frame_id}_pred_label_{ind + 1}.png"
            full_path = os.path.join(test_save_path, filename)
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            plt.close('all')
        # get resolution
        case_raw = r"D:\2024\ssr\DATA\ACDC_or\database\training" + case+ '.nii.gz'
        case_raw = r"D:\2024\ssr\DATA\ACDC_or\database\training\patient001\patient001_frame01_gt.nii.gz"
        case_raw = sitk.ReadImage(case_raw)
        raw_spacing = case_raw.GetSpacing()
        raw_spacing_new = []
        raw_spacing_new.append(raw_spacing[2])
        raw_spacing_new.append(raw_spacing[1])
        raw_spacing_new.append(raw_spacing[0])
        raw_spacing = raw_spacing_new

    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase_nan(prediction == i, label == i,raw_spacing))

    # if test_save_path is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
    #     #sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
    #     #sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    #     print('saved successfully!')
    return metric_list


def test_single_image(sampled_batch, net, classes, multimask_output, i, metric, patch_size=[256, 256],
                      input_size=[224, 224], test_save_path=None):
    image, target = sampled_batch['image'], sampled_batch['label']
    image, target = image.cuda(), target.cuda()

    if len(image.shape) != 3:
        net.eval()
        # custom_mask = AdaptiveMaskNet()
        # custom_mask.load_state_dict(torch.load('output/CXR_1024new_custom_maskatt/iter_600mask.pth', map_location='cuda'))
        # custom_mask = custom_mask.cuda()
        # custom_mask.eval()
        with torch.no_grad():
            outputs = net(image, multimask_output, patch_size[0])
            output_masks1 = outputs['masks']
            output_masks2 = outputs['masks2']
            output_masks1=torch.softmax(output_masks1, dim=1)
            output_masks2 = torch.softmax(output_masks2, dim=1)

            output_masks=custom_mask(output_masks1,  output_masks2,i)
            output_masks=torch.softmax(output_masks, dim=1)
            out = (output_masks1+output_masks2)*0.5
            metric.update(output_masks, target)

            pred = torch.max(output_masks, dim=1)[1]
            pred1 = torch.max(output_masks1, dim=1)[1]
            pred2 = torch.max(output_masks2, dim=1)[1]

            _, axs = plt.subplots(1, 2, figsize=(25, 25))
            axs[0].imshow(pred.cpu().permute(1, 2, 0).detach().numpy(), cmap='gray')
            axs[1].imshow(target.cpu().permute(1, 2, 0).detach().numpy(), cmap='gray')
            plt.subplots_adjust(wspace=0.01, hspace=0)
            filename = f"./test{i + 1}.png"
            full_path = os.path.join(test_save_path, filename)
            plt.savefig(full_path, bbox_inches="tight", dpi=300)
            plt.close('all')

            predname = f"./pred{i + 1}.png"
            pred_path = os.path.join(test_save_path, predname)
            pred_image = pred.cpu().permute(1, 2, 0).detach().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(pred_image, cmap='gray', interpolation='none')
            plt.axis('off')
            plt.savefig(pred_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
            plt.close('all')
            #
            # pred1name = f"./pred{i + 1}_1.png"
            # pred1_path = os.path.join(test_save_path, pred1name)
            # pred1_image = pred1.cpu().permute(1, 2, 0).detach().numpy()
            # plt.figure(figsize=(10, 10))
            # plt.imshow(pred1_image, cmap='gray', interpolation='none')
            # plt.axis('off')
            # plt.savefig(pred1_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
            #
            # pred2name = f"./pred{i + 1}_2.png"
            # pred2_path = os.path.join(test_save_path, pred2name)
            # pred2_image = pred2.cpu().permute(1, 2, 0).detach().numpy()
            # plt.figure(figsize=(10, 10))
            # plt.imshow(pred2_image, cmap='gray', interpolation='none')
            # plt.axis('off')
            # plt.savefig(pred2_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
            #
            # plt.close('all')



def test_single_image_mean(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        slice = image#[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0, patch_size[0], patch_size[1]
        inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction = pred
    else:
        x, y = image.shape[-2:]
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks1 = outputs['masks']
            output_masks2 = outputs['masks2']
            output_masks1 = torch.softmax(output_masks1, dim=1)
            output_masks2 = torch.softmax(output_masks2, dim=1)
            output_masks = (output_masks1 + output_masks2)/2.0
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase_nospacing(prediction == i, label == i))

    if test_save_path is not None:
        # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        print('saved successfully!')
    return metric_list

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

import torch
import torch.nn.functional as F

# def cosine_similarity(x, y):
#     """
#     计算两个张量的余弦相似度
#     """
#     x_flat = x.view(x.size(0), -1)  # 将每个图片展平成一个向量
#     y_flat = y.view(y.size(0), -1)  # 同上
#     sim = F.cosine_similarity(x_flat, y_flat, dim=1)
#     return sim
#
# def custom_mask(outputs1, outputs2, kernel_size=11, similarity_threshold=0.9):
#     # 1. 计算余弦相似度
#     cos_sim = cosine_similarity(outputs1, outputs2)  # 计算outputs1和outputs2的余弦相似度
#     print("Cosine Similarity:", cos_sim)  # 打印余弦相似度，可以去掉
#
#     # 2. 根据余弦相似度生成掩码
#     # 如果相似度大于阈值，则认为它们相似，使用均值；否则使用均值滤波
#     threshold_mask = cos_sim > similarity_threshold
#
#     # 3. 对于相似的部分，使用均值
#     fused_output = (outputs1 + outputs2) / 2.0  # 计算输出的均值
#
#     # 4. 对于不相似的部分，应用均值滤波
#     kernel_size = int(kernel_size)  # 获取卷积核大小
#     kernel_size = max(kernel_size, 3)  # 最小卷积核大小为3
#
#     # 定义均值滤波核
#     kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
#     kernel = kernel.to(outputs1.device)  # 确保卷积核在正确的设备上
#     padding = (kernel_size - 1) // 2  # 为了保持原始尺寸，需要相应的填充
#     kernel = kernel.expand(fused_output.size(1), -1, -1, -1)  # 扩展卷积核的维度以适应 F.conv2d
#
#     # 执行卷积操作，获得均值滤波后的输出
#     aver_vector = F.conv2d(fused_output, kernel, padding=padding, groups=fused_output.size(1))
#
#     # 5. 根据余弦相似度掩码，选择使用均值或均值滤波
#     final_output = torch.where(threshold_mask.view(-1, 1, 1, 1), fused_output, aver_vector)
#
#     return final_output

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class AdaptiveMaskNet(nn.Module):
#     def __init__(self, initial_threshold=0.5, initial_kernel_size=11):
#         super(AdaptiveMaskNet, self).__init__()
#         # 使用可学习的阈值和卷积核大小，并确保它们是浮点数类型
#         self.threshold = nn.Parameter(torch.tensor(initial_threshold, dtype=torch.float32))  # 可学习的阈值
#         self.kernel_size = nn.Parameter(torch.tensor(initial_kernel_size, dtype=torch.float32))  # 可学习的卷积核大小
#
#     def forward(self, outputs1, outputs2):
#         # 计算整个输出的差异
#         entropy1 = -torch.sum(outputs1 * torch.log(outputs1 + 1e-10), dim=1, keepdim=True)
#         entropy2 = -torch.sum(outputs2 * torch.log(outputs2 + 1e-10), dim=1, keepdim=True)
#         # 计算熵的差异
#         entropy_diff = torch.abs(entropy1 - entropy2)
#         entropy_mask = entropy_diff < self.threshold
#
#
#
#
#         diff = torch.abs(outputs1 - outputs2)
#         diff=torch.softmax(diff,dim=1)
#         # 使用可学习的阈值生成阈值掩码
#         threshold_mask = diff < self.threshold
#
#
#         heatmap = threshold_mask[:, 1, :, :].detach().cpu().numpy()  # 从GPU移动到CPU并转换为NumPy数组
#         heatmap = np.mean(heatmap, axis=0)
#
#         heatmap = np.maximum(heatmap, 0)
#         heatmap /= np.max(heatmap)
#
#         plt.matshow(heatmap, cmap=plt.cm.coolwarm)
#         plt.show()
#         # 计算outputs1和outputs2的平均值
#         average = (outputs1 + outputs2) / 2
#
#         # 对小于阈值的部分填充平均值，否则填充0
#         new_vector = torch.where(threshold_mask, average, torch.zeros_like(outputs1))
#
#         # 计算反向掩码
#         inverse_threshold_mask = ~threshold_mask
#
#         # 动态调整卷积核大小
#         kernel_size = int(self.kernel_size.item())  # 获取卷积核大小
#         kernel_size = max(kernel_size, 3)  # 最小卷积核大小为3
#
#         # 定义均值滤波核
#         kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
#         kernel = kernel.cuda()
#         # 为了保持原始尺寸，需要相应的填充
#         padding = (kernel_size - 1) // 2
#
#         # 扩展卷积核的维度以适应 F.conv2d
#         kernel = kernel.expand(new_vector.size(1), -1, -1, -1)  # 适配输入张量的通道数
#
#         # 执行卷积操作
#         aver_vector = F.conv2d(new_vector, kernel, padding=padding, groups=new_vector.size(1))
#
#         # 使用inverse_threshold_mask过滤aver_vector
#         filtered_aver_vector = torch.where(inverse_threshold_mask, aver_vector, torch.zeros_like(new_vector))
#
#         # 将过滤后的aver_vector与new_vector相加
#         final_vector = new_vector + filtered_aver_vector
#         return final_vector
def fourier_transform(x):
    """
    对输入的图像张量进行傅里叶变换，并将频谱中心化。

    :param x: 输入的图像张量，形状为 [batch_size, channels, height, width]
    :return: 傅里叶变换后的频域图像，形状为 [batch_size, channels, height, width]
    """
    # 对每个通道进行傅里叶变换
    fft_image = torch.fft.fft2(x)  # 对图像进行傅里叶变换
    fft_image = torch.fft.fftshift(fft_image)  # 将频谱中心化
    return fft_image

def visualize_and_save_frequency_domain(fft,enhanced_image_real, save_path,i, prefix=""):
    # 可视化频域特征
    fft_magnitude = torch.abs(fft)

    # 转换为 NumPy 格式以便可视化
    fft_magnitude_np = fft_magnitude[0, 0].cpu().numpy()
    enhanced_image_real_np =  enhanced_image_real[0, 0].cpu().numpy()
    enhanced_image_real_np = (enhanced_image_real_np - enhanced_image_real_np.min()) / (
                enhanced_image_real_np.max() - enhanced_image_real_np.min())

    # 保存 FFT Magnitude
    plt.figure(figsize=(8, 8))
    plt.imshow(np.log(fft_magnitude_np + 1), cmap='viridis')
    plt.axis('off')
    fft_save_path = os.path.join(save_path, f"{i}_fft{prefix}.png")
    plt.savefig(fft_save_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(enhanced_image_real_np, cmap='viridis')
    plt.axis('off')
    enhanced_image_real_save_path = os.path.join(save_path, f"{i}_e{prefix}.png")
    plt.savefig(enhanced_image_real_save_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
    plt.close()

def custom_mask(outputs1, outputs2, i,kernel_size=3):
    # 2. 进行频域处理：对输出进行傅里叶变换提取频域特征
    fft1 = fourier_transform(outputs1)
    fft2 = fourier_transform(outputs2)

    # 设定低频和高频的掩码
    low_freq_mask1 = torch.zeros_like(fft1)
    low_freq_mask1[:, :, fft1.shape[2] // 4:fft1.shape[2] // 2, fft1.shape[3] // 4:fft1.shape[3] // 2] = 1

    high_freq_mask1 = torch.ones_like(fft1)
    high_freq_mask1[:, :, fft1.shape[2] // 2:fft1.shape[2] // 4 * 3, fft1.shape[3] // 2:fft1.shape[3] // 4 * 3] = 0

    low_freq_mask2 = torch.zeros_like(fft2)
    low_freq_mask2[:, :, fft2.shape[2] // 4:fft2.shape[2] // 2, fft2.shape[3] // 4:fft2.shape[3] // 2] = 1

    high_freq_mask2 = torch.ones_like(fft2)
    high_freq_mask2[:, :, fft2.shape[2] // 2:fft2.shape[2] // 4 * 3, fft2.shape[3] // 2:fft2.shape[3] // 4 * 3] = 0

    # 获取低频部分
    low_freq1 = fft1 * low_freq_mask1
    low_freq2 = fft2 * low_freq_mask2

    # 获取高频中的低频部分
    high_freq_low_freq1 = fft1 * high_freq_mask1 * low_freq_mask1
    high_freq_low_freq2 = fft2 * high_freq_mask2 * low_freq_mask2

    # 合成：低频 * (高频中的低频) + 低频
    enhanced_fft1 = low_freq1 + 0.5*high_freq_low_freq1
    enhanced_fft2 = low_freq2 + 0.5*high_freq_low_freq2

    # 将频域转换回图像空间
    enhanced_fft1_shifted = torch.fft.ifftshift(enhanced_fft1)
    enhanced_image1 = torch.fft.ifft2(enhanced_fft1_shifted)
    enhanced_image_real1 = 0.5*torch.real(enhanced_image1)+outputs1

    enhanced_fft2_shifted = torch.fft.ifftshift(enhanced_fft2)
    enhanced_image2 = torch.fft.ifft2(enhanced_fft2_shifted)
    enhanced_image_real2 = 0.5*torch.real(enhanced_image2)+outputs2

    # 结合两种增强后的图像
    enhanced_fft = 0.5 * (enhanced_image_real1 + enhanced_image_real2)

    # 卷积操作以保持图像大小
    kernel_size = int(kernel_size)  # 获取卷积核大小
    kernel_size = max(kernel_size, 7)  # 最小卷积核大小为3
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
    kernel = kernel.to(outputs1.device)

    padding = (kernel_size - 1) // 2
    kernel = kernel.expand(enhanced_fft.size(1), -1, -1, -1)

    # # 执行卷积操作，确保输出尺寸与原图一致
    # save_path=r"D:\2024\ssr\MeanFilter"
    # pred= torch.max(enhanced_fft, dim=1)[1]
    # predname = f"./pred{i + 1}.png"
    # pred_path = os.path.join(save_path, predname)
    # pred_image = pred.cpu().permute(1, 2, 0).detach().numpy()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(pred_image, cmap='gray', interpolation='none')
    # plt.axis('off')
    # plt.savefig(pred_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
    # plt.close('all')





    aver_vector = F.conv2d(enhanced_fft, kernel, padding=padding, groups=enhanced_fft.size(1))

    save_path = "./resault/CXR/new/600"
    # visualize_and_save_frequency_domain(fft1, enhanced_image_real1,save_path,i+1, prefix="1")
    # visualize_and_save_frequency_domain(fft2,  enhanced_image_real2,save_path,i+1, prefix="2")
    return aver_vector


# def custom_mask(outputs1, outputs2,kernel_size=3):
#
#
#     # 2. 进行频域处理：对融合后的输出进行傅里叶变换提取频域特征
#     fft1 = fourier_transform(outputs1)
#     fft2 = fourier_transform(outputs2)
#     low_freq_mask1 = torch.zeros_like(fft1)
#     low_freq_mask1[:, :, fft1.shape[2] // 4:fft1.shape[2] // 2,fft1.shape[3] // 4:fft1.shape[3] // 2] = 1
#     enhanced_fft_magnitude1 = fft1 * low_freq_mask1
#     enhanced_fft_shifted1 = torch.fft.ifftshift(enhanced_fft_magnitude1)
#     enhanced_image1 = torch.fft.ifft2(enhanced_fft_shifted1)
#     enhanced_image_real1 = 0*torch.real(enhanced_image1)+outputs1
#
#     low_freq_mask2 = torch.zeros_like(fft2)
#     low_freq_mask2[:, :, fft1.shape[2] // 4:fft1.shape[2] // 2, fft1.shape[3] // 4:fft1.shape[3] // 2] = 1
#     enhanced_fft_magnitude2 = fft1 * low_freq_mask2
#     enhanced_fft_shifted2 = torch.fft.ifftshift(enhanced_fft_magnitude2)
#     enhanced_image2 = torch.fft.ifft2(enhanced_fft_shifted2)
#     enhanced_image_real2 = 0*torch.real(enhanced_image2)+outputs2
#
#     enhanced_fft=0.5*(enhanced_image_real1+enhanced_image_real2)
#
#     kernel_size = int(kernel_size)  # 获取卷积核大小
#     kernel_size = max(kernel_size, 3)  # 最小卷积核大小为3
#
#     kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
#     kernel = kernel.to(outputs1.device)
#
#     padding = (kernel_size - 1) // 2
#
#     kernel = kernel.expand(enhanced_fft.size(1), -1, -1, -1)
#     # 执行卷积操作
#     aver_vector = F.conv2d( enhanced_fft, kernel, padding=padding, groups=enhanced_fft.size(1))

    # test_save_path = "./resault/CXR/new/iter_300"
    # fff1name = f"./ff{i + 1}_1.png"
    # ff1_path = os.path.join(test_save_path, fff1name)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(torch.log(torch.abs(fft1[0, 1, :, :])).cpu().detach().numpy(),cmap='jet')
    # plt.axis('off')
    # plt.savefig(ff1_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
    #
    # ff2name = f"./ff{i + 1}_2.png"
    # ff2_path = os.path.join(test_save_path, ff2name)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(torch.log(torch.abs(fft2[0, 1, :, :])).cpu().detach().numpy(),cmap='jet')
    # plt.axis('off')
    # plt.savefig(ff2_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
    #
    #
    # # 可视化增强后的图像
    # e1name = f"./e{i + 1}_1.png"
    # e1_path = os.path.join(test_save_path, e1name)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(enhanced_image_real1[0, 1, :, :].cpu().detach().numpy())
    # plt.axis('off')
    # plt.savefig(e1_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
    #
    #
    # e2name = f"./e{i + 1}_2.png"
    # e2_path = os.path.join(test_save_path, e2name)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(enhanced_image_real2[0, 1, :, :].cpu().detach().numpy())
    # plt.axis('off')
    # plt.savefig(e2_path, bbox_inches='tight', pad_inches=0, dpi=300, format='png')
    #
    #
    #
    #
    #
    #
    # plt.close('all')

    # return aver_vector






# def adaptive_threshold_kernel(outputs1, outputs2):
#     # 计算输出之间的差异
#     diff = torch.abs(outputs1 - outputs2)
#
#     # 计算差异的均值和标准差
#     diff_mean = torch.mean(diff)
#     diff_std = torch.std(diff)
#
#     # 动态计算阈值
#     threshold = diff_mean + diff_std  # 使用均值 + 标准差作为阈值
#
#     # 动态计算卷积核大小（可以根据图像大小或方差来决定）
#     height, width = outputs1.shape[2], outputs1.shape[3]
#     kernel_size = int(min(height, width) * 0.05)  # 卷积核大小是图像尺寸的 5%
#     kernel_size = max(kernel_size, 3)  # 最小卷积核大小为3
#
#     return threshold, kernel_size
#
#
# def custom_mask(outputs1, outputs2, threshold=None, kernel_size=None):
#     if threshold is None or kernel_size is None:
#         threshold, kernel_size = adaptive_threshold_kernel(outputs1, outputs2)
#
#     # 计算整个输出的差异
#     diff = torch.abs(outputs1 - outputs2)
#     threshold_mask = diff < threshold
#
#     # 计算outputs1和outputs2的平均值
#     average = (outputs1 + outputs2) / 2
#
#     new_vector = torch.where(threshold_mask, average, torch.zeros_like(outputs1))
#     inverse_threshold_mask = ~threshold_mask
#
#     kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
#     kernel = kernel.cuda()
#     padding = (kernel_size - 1) // 2
#
#     kernel = kernel.expand(new_vector.size(1), -1, -1, -1)
#     aver_vector = F.conv2d(new_vector, kernel, padding=padding, groups=new_vector.size(1))
#
#     filtered_aver_vector = torch.where(inverse_threshold_mask, aver_vector, torch.zeros_like(new_vector))
#
#     final_vector = new_vector + filtered_aver_vector
#     return final_vector


# def custom_mask(outputs1, outputs2, threshold=0.9,kernel_size=11):
#     # 计算整个输出的差异
#     diff = torch.abs(outputs1 - outputs2)
#     # 创建一个阈值掩码，其中差异小于阈值的位置为True
#     diff=torch.sigmoid(diff)
#     threshold_mask = diff < threshold
#
#
#     # 计算outputs1和outputs2的平均值
#     average = (outputs1 + outputs2) / 2
#
#     # 首先，对小于阈值的部分填充平均值，否则填充0
#     new_vector = torch.where(threshold_mask, average, torch.zeros_like(outputs1))
#
#     # 计算反向掩码
#     inverse_threshold_mask = ~threshold_mask
#
#     # 对整个new_vector进行均值滤波，这里假设使用简单的均值滤波
#     # 需要一个窗口大小，这里假设为3x3，并且假设outputs1是4D张量（批次，通道，高度，宽度）
#
#     # 定义均值滤波核
#     kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
#     kernel = kernel.cuda()
#     # 为了保持原始尺寸，需要相应的填充
#     padding = (kernel_size - 1) // 2
#
#     # 扩展卷积核的维度以适应 F.conv2d
#     kernel = kernel.expand( new_vector.size(1), -1, -1, -1)  # 适配输入张量的通道数
#
#     # 执行卷积操作
#     aver_vector = F.conv2d( new_vector, kernel, padding=padding, groups= new_vector.size(1))
#
#     # 使用inverse_threshold_mask过滤aver_vector
#     filtered_aver_vector = torch.where(inverse_threshold_mask, aver_vector, torch.zeros_like(new_vector))
#
#     # 将过滤后的aver_vector与new_vector相加
#     final_vector = new_vector + filtered_aver_vector
#     return final_vector



def  calculate_dice(pred, label):
    pred1=pred.clone()
    label1=label.clone()
    pred1 = pred1.detach() .cpu().numpy()
    label1 = label1.detach().cpu().numpy()
    intersection = np.sum(pred1 * label1)
    dice = (2. * intersection) / (np.sum(pred1) + np.sum(label1) + 1e-8)  # 避免分母为零
    return dice


def calculate_entropy(pred):
    pred1=pred.clone()

    pred1 = pred1.detach().cpu().numpy()

    pred1 = np.clip(pred1, 1e-8, 1 - 1e-8)  # 避免 log(0)
    return entropy(pred1.flatten())  # 熵值计算

def calculate_score(windows,  pred, label,labeled_bs):

    score1=calculate_dice(pred[:labeled_bs], label[:labeled_bs].unsqueeze(1))
    score2= calculate_entropy(pred[labeled_bs:])
    score=score1+score2

    return score
