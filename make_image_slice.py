import os
import numpy as np
import torch
import glob
import nibabel as nib
def image_preprocess(image):
    t1_clipped = np.clip(
                        image,
                        np.quantile(image, 0.001),
                        np.quantile(image, 0.999),
                    )
    t1_normalized = (t1_clipped - np.min(t1_clipped)) / (
        np.max(t1_clipped) - np.min(t1_clipped)
    )

    return t1_normalized

data_dir = '/hpc/data/home/bme/yubw/taotl/BraTS-2023_challenge/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training/'
subject_dirs = sorted([os.path.join(data_dir, name) for name in os.listdir(data_dir)])
print(subject_dirs)
for idx in range(len(subject_dirs)):
    subject_dir = subject_dirs[idx]
    print(subject_dir)
    image_path = glob.glob(os.path.join(subject_dir, '*t1n.nii.gz'))[0]
    cropped_image_path = glob.glob(os.path.join(subject_dir, '*t1n-voided.nii.gz'))[0]
    healthy_mask_path = glob.glob(os.path.join(subject_dir, '*healthy.nii.gz'))[0]
    unhealthy_mask_path = glob.glob(os.path.join(subject_dir, '*unhealthy.nii.gz'))[0]


    # 加载图像和掩膜
    image = nib.load(image_path).get_fdata().astype(
                    np.float32
                )
    healthy_mask = nib.load(healthy_mask_path).get_fdata().astype(
                    np.float32
                )
    unhealthy_mask = nib.load(unhealthy_mask_path).get_fdata().astype(
                    np.float32
                )
    cropped_image = nib.load(cropped_image_path).get_fdata().astype(
                    np.float32
                )
    image = image_preprocess(image)
    cropped_image = image_preprocess(cropped_image)
    mask_affine = nib.load(healthy_mask_path).affine
    # image = np.pad(image, ((0, 0), (0, 0), (3, 2)), mode='constant')
    # healthy_mask = np.pad(healthy_mask, ((0, 0), (0, 0), (3, 2)), mode='constant')
    # cropped_image = np.pad(cropped_image, ((0, 0), (0, 0), (3, 2)), mode='constant')
    # unhealthy_mask = np.pad(unhealthy_mask, ((0, 0), (0, 0), (3, 2)), mode='constant')

    nonzero_coords = np.nonzero(healthy_mask)
    center_x = (np.min(nonzero_coords[0]) + np.max(nonzero_coords[0])) // 2
    center_y = (np.min(nonzero_coords[1]) + np.max(nonzero_coords[1])) // 2
    center_z = (np.min(nonzero_coords[2]) + np.max(nonzero_coords[2])) // 2
    image_shape = [240,240,155]
    # 计算裁剪区域的边界
    crop_x1 = max(center_x - 48, 0)
    crop_x2 = min(center_x + 48, image_shape[0])
    crop_y1 = max(center_y - 48, 0)
    crop_y2 = min(center_y + 48, image_shape[1])
    crop_z1 = max(center_z - 48, 0)
    crop_z2 = min(center_z + 48, image_shape[2])

    # 如果裁剪区域小于 96x96x96,则在另一边扩展
    crop_size_x = crop_x2 - crop_x1
    crop_size_y = crop_y2 - crop_y1
    crop_size_z = crop_z2 - crop_z1

    if crop_size_x < 96:
        if center_x - 48 < 0:
            crop_x1 = 0
            crop_x2 = 96
        else:
            crop_x1 = image_shape[0] - 96
            crop_x2 = image_shape[0]

    if crop_size_y < 96:
        if center_y - 48 < 0:
            crop_y1 = 0
            crop_y2 = 96
        else:
            crop_y1 = image_shape[1] - 96
            crop_y2 = image_shape[1]

    if crop_size_z < 96:
        if center_z - 48 < 0:
            crop_z1 = 0
            crop_z2 = 96
        else:
            crop_z1 = image_shape[2] - 96
            crop_z2 = image_shape[2]

    image = image[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    healthy_mask = healthy_mask[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    cropped_image = cropped_image[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    unhealthy_mask = unhealthy_mask[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]


    # 二值化掩膜
    unhealthy_mask = np.where(unhealthy_mask > 0, 1, 0)
    unhealthy_mask_bool = (unhealthy_mask == 1)

    # 使用掩码对图像进行裁剪
    image[unhealthy_mask_bool] = 0

    image = image[np.newaxis, ...]
    cropped_image = cropped_image[np.newaxis, ...]
    healthy_mask = healthy_mask[np.newaxis, ...]
    # print(image.shape)

    # save_dir = os.path.join(subject_dir, 'preprocessed_data')
    # os.makedirs(save_dir, exist_ok=True)

    # 保存每个slice的数据
    for z in range(image.shape[3]):
        slice_image = image[:, :, :, z]
        slice_healthy_mask = healthy_mask[:, :, :, z]
        slice_cropped_image = cropped_image[:, :, :, z]
        # slice_unhealthy_mask = unhealthy_mask[:, :, z]
        
        # 保存为npy文件
        np.savez(os.path.join(subject_dir, f'slice_{z}.npz'), 
                 image=slice_image, 
                 healthy_mask=slice_healthy_mask,
                 cropped_image=slice_cropped_image)
        