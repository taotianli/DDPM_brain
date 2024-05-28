from PIL import Image
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
# import torchvision
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import glob
# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# from torchmetrics.regression import (
#     MeanAbsoluteError,
#     MeanSquaredError,
#     MeanSquaredLogError,
# )
# from inpainting.challenge_metrics_2023 import generate_metrics, read_nifti_to_tensor

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([image for image in images.cpu()], dim=-1)
        ], dim=-2).permute(1,2,0).cpu())
    plt.show()

def save_images(img, c_img, dd_img, d_img, path, **kwargs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.figure(figsize=(8, 8))
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Ground-truth')
    axes[0,1].imshow(c_img, cmap='gray')
    axes[0,1].set_title('Cropped guidance')
    axes[1,0].imshow(dd_img, cmap='gray')
    axes[1,0].set_title('DDPM genarated')
    axes[1,1].imshow(d_img, cmap='gray')
    axes[1,1].set_title('Final infilled image')
    plt.savefig(path)

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

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, train=True, device='cuda'):
        self.data_dir = data_dir
        self.train = train
        self.device = device
        self.subject_dirs = sorted(glob.glob(os.path.join(data_dir, '**', '96*.npz'), recursive=True))
        train_size = int(0.95*len(self.subject_dirs))
        if self.train:
            self.subject_dirs = self.subject_dirs[:train_size]
        else:
            self.subject_dirs = self.subject_dirs[train_size:]
        # print(self.subject_dirs)

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_slice_path = self.subject_dirs[idx]
        with np.load(subject_slice_path) as data:
            image = data['image']
            healthy_mask = data['healthy_mask']
            cropped_image = data['cropped_image']

        # image 是健康的图像，即只抠掉肿瘤区域的图像 cropped_image是抠掉要生成区域的图像，mask是健康图像的掩膜
        image = torch.from_numpy(image).squeeze(-1)
        cropped_image = torch.from_numpy(cropped_image).squeeze(-1)
        healthy_mask = torch.from_numpy(healthy_mask).squeeze(-1)

        return image.to(self.device), cropped_image.to(self.device), healthy_mask.to(self.device), subject_slice_path


class BrainTumorDataset_inference(Dataset):
    def __init__(self, data_dir, train=True, device='cuda'):
        self.data_dir = data_dir
        self.train = train
        self.device = device
        self.subject_dirs = sorted(glob.glob(os.path.join(data_dir, '**', '96*.npz'), recursive=True))
        self.subject_dirs = self.subject_dirs[:]

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_slice_path = self.subject_dirs[idx]
        with np.load(subject_slice_path) as data:
            image = data['image']
            healthy_mask = data['healthy_mask']
            cropped_image = data['cropped_image']

        image = torch.from_numpy(image).squeeze(-1)
        cropped_image = torch.from_numpy(cropped_image).squeeze(-1)# cropped image 就是 labels
        healthy_mask = torch.from_numpy(healthy_mask).squeeze(-1)

        return image.to(self.device), cropped_image.to(self.device), healthy_mask.to(self.device), subject_slice_path
    

class BrainTumorDataset_validation(Dataset):
    def __init__(self, data_dir, train=True, device='cuda'):
        self.data_dir = data_dir
        self.train = train
        self.device = device
        self.subject_dirs = sorted(glob.glob(os.path.join(data_dir, '**', '96*.npz'), recursive=True))

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_slice_path = self.subject_dirs[idx]
        with np.load(subject_slice_path) as data:
            healthy_mask = data['healthy_mask']
            cropped_image = data['cropped_image']

        cropped_image = torch.from_numpy(cropped_image).squeeze(-1)
        healthy_mask = torch.from_numpy(healthy_mask).squeeze(-1)

        return cropped_image.to(self.device), healthy_mask.to(self.device)


def get_data(args):
    dataset = BrainTumorDataset(args.dataset_path, args.train, device=args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    return dataloader

def get_data_inference(args):
    dataset = BrainTumorDataset_inference(args.dataset_path, args.train, device=args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    return dataloader

def get_data_valid(args):
    dataset = BrainTumorDataset_validation(args.dataset_path, args.train, device=args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader


def setup_logging(run_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{run_name}', exist_ok=True)
    os.makedirs(f'models/{run_name}', exist_ok=True)
