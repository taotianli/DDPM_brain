from PIL import Image
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import glob

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([image for image in images.cpu()], dim=-1)
        ], dim=-2).permute(1,2,0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.subject_dirs = sorted([os.path.join(data_dir, name) for name in os.listdir(data_dir)])
        # print(self.subject_dirs)

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]
        image_path = glob.glob(os.path.join(subject_dir, '*t1n.nii.gz'))[0]
        mask_path = glob.glob(os.path.join(subject_dir, '*seg.nii.gz'))[0]

        # 加载图像和掩膜
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        mask_affine = nib.load(mask_path).affine
        image = np.pad(image, ((0, 0), (0, 0), (3, 2)), mode='constant')
        mask = np.pad(mask, ((0, 0), (0, 0), (3, 2)), mode='constant')
        # print(mask_affine)


        # 二值化掩膜
        mask = np.where(mask > 0, 1, 0)
        mask_bool = (mask == 1)

        # 使用掩码对图像进行裁剪
        cropped_image = image.copy()
        cropped_image[mask_bool] = 0

        image = image[np.newaxis, ...]
        cropped_image = cropped_image[np.newaxis, ...]
        mask = mask[np.newaxis, ...]


        return image, cropped_image, mask, mask_affine

# 数据加载
# train_dataset = BrainTumorDataset('D:\\BraTS\\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
# dataset = BrainTumorDataset('D:\\BraTS\\ASNR_test')
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

def get_data(args):
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(80),
    #     torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8,1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset = BrainTumorDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{run_name}', exist_ok=True)
    os.makedirs(f'models/{run_name}', exist_ok=True)