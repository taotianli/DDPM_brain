import os
import numpy as np
import torch
import glob
import nibabel as nib

data_dir = '/hpc/data/home/bme/taotl/DDPM_brain/data/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training/'
subject_dirs = sorted([os.path.join(data_dir, name) for name in os.listdir(data_dir)])
# print(subject_dirs)
for idx in range(len(subject_dirs)):
    
    subject_dir = subject_dirs[idx]
    subject_name = os.path.split(subject_dir)[1]
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
    mask_affine = nib.load(healthy_mask_path).affine

    image_mask = np.where(image > 0, 1, 0)
    image_mask_bool = (image_mask != 1) # 没有组织的地方设为0
    healthy_mask[image_mask_bool] = 0.
    
    img = nib.Nifti1Image(healthy_mask, mask_affine)

    nib.save(img, os.path.join(data_dir, subject_name, subject_name + '-mask-healthy-modefied.nii.gz'))



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


subject_dirs = sorted([os.path.join(data_dir, name) for name in os.listdir(data_dir)])
for idx in range(len(subject_dirs)):
    subject_dir = subject_dirs[idx]
    # print(subject_dir)
    image_path = glob.glob(os.path.join(subject_dir, '*t1n.nii.gz'))[0]
    cropped_image_path = glob.glob(os.path.join(subject_dir, '*t1n-voided.nii.gz'))[0]
    healthy_mask_path = glob.glob(os.path.join(subject_dir, '*healthy-modefied.nii.gz'))[0]
    unhealthy_mask_path = glob.glob(os.path.join(subject_dir, '*unhealthy.nii.gz'))[0]


    # 加载图像和掩膜
    image = nib.load(image_path).get_fdata().astype(
                    np.float32
                )
    image_copy = nib.load(image_path).get_fdata().astype(
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

    nonzero_coords = np.nonzero(healthy_mask)
    center_x = (np.min(nonzero_coords[0]) + np.max(nonzero_coords[0])) // 2
    center_y = (np.min(nonzero_coords[1]) + np.max(nonzero_coords[1])) // 2
    center_z = (np.min(nonzero_coords[2]) + np.max(nonzero_coords[2])) // 2
    image_shape = [240,240,155]
    img_size = 96
    # # 计算裁剪区域的边界
    crop_x1 = max(center_x - int(img_size/2), 0)
    crop_x2 = min(center_x + int(img_size/2), image_shape[0])
    crop_y1 = max(center_y - int(img_size/2), 0)
    crop_y2 = min(center_y + int(img_size/2), image_shape[1])
    crop_z1 = max(center_z - 48, 0)
    crop_z2 = min(center_z + 48, image_shape[2])
    crop_z1 = np.min(nonzero_coords[2])
    crop_z2 = np.max(nonzero_coords[2])

    # 如果裁剪区域小于 96x96x96,则在另一边扩展
    crop_size_x = crop_x2 - crop_x1
    crop_size_y = crop_y2 - crop_y1
    crop_size_z = crop_z2 - crop_z1
    #保存几何坐标信息
    # print(crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2)
    geometric_list = [crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2]

    if crop_size_x < img_size:
        if center_x - int(img_size/2) < 0:
            crop_x1 = 0
            crop_x2 = img_size
        else:
            crop_x1 = image_shape[0] - int(img_size)
            crop_x2 = image_shape[0]

    if crop_size_y < img_size:
        if center_y - int(img_size/2) < 0:
            crop_y1 = 0
            crop_y2 = img_size
        else:
            crop_y1 = image_shape[1] - int(img_size)
            crop_y2 = image_shape[1]



    image = image[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    healthy_mask = healthy_mask[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    cropped_image = cropped_image[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]
    unhealthy_mask = unhealthy_mask[crop_x1:crop_x2, crop_y1:crop_y2, crop_z1:crop_z2]


    # 二值化掩膜
    unhealthy_mask = np.where(unhealthy_mask > 0, 1, 0)
    unhealthy_mask_bool = (unhealthy_mask == 1)
    healthy_mask = np.where(healthy_mask > 0, 1, 0)
    healthy_mask_bool = (healthy_mask == 1)

    # 使用掩码对图像进行裁剪
    image[unhealthy_mask_bool] = 0.

    image = image[np.newaxis, ...]
    cropped_image = cropped_image[np.newaxis, ...]
    healthy_mask = healthy_mask[np.newaxis, ...]


    # 保存每个slice的数据
    counter = 0
    for z in range(image.shape[3]):
        slice_healthy_mask = healthy_mask[:, :, :, z]
        slice_image = image[:, :, :, z]
        slice_cropped_image = cropped_image[:, :, :, z]
        if np.sum(slice_healthy_mask) > 20:
            # 保存为npy文件
            np.savez(os.path.join(subject_dir, str(img_size) + f'_slice_{z}.npz'), 
                     image=slice_image, 
                     healthy_mask=slice_healthy_mask,
                     cropped_image=slice_cropped_image)
        else:
            counter += 1


            # plt.imshow(slice_image[0,:,:], cmap='gray')
            # plt.show()
