import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
# from utils import get_data_inference
from modules import UNet_conditional, EMA, UNet_conditional_concat, UNet_conditional_fully_concat, UNet_conditional_fully_add
from modules import UNet_conditional_concat_with_mask, UNet_conditional_concat_with_mask_v2, UNet_conditional_concat_Large
from modules import UNet_conditional_concat_XLarge
import logging
from torch.utils.tensorboard import SummaryWriter
# from utils import _structural_similarity_index, _peak_signal_noise_ratio, _mean_squared_error

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
"""
需要修改的地方：

"""

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=240, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def _timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, masks, cfg_scale=0):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device) + 0.5*labels
            # x = labels+masks
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # predicted_noise = model(x, t, labels)
                # print(labels.shape, masks.shape)
                predicted_noise = model(x, t, labels, masks)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x

def inpaint_image(original_image, generated_image, mask):
    """
    将生成的图像融合到原始图像的指定区域中。
    
    参数:
    original_image (torch.Tensor): 原始图像
    generated_image (torch.Tensor): 生成的图像
    mask (torch.Tensor): 掩码图像, 1表示需要inpaint的区域, 0表示保留原图
    
    返回:
    torch.Tensor: 输出的合成图像
    """
    # 将三个输入tensor转换到相同的设备上
    device = original_image.device
    mask = mask.to(device)
    generated_image = generated_image.to(device)
    
    # 使用掩码融合原图和生成的图像
    output_image = original_image.clone()
    # print(output_image.shape, mask.shape, generated_image.shape)
    output_image = output_image * (1 - mask) + generated_image * mask
    
    return output_image

def inpaint_image(original_image, generated_image, mask):
    """
    将生成的图像融合到原始图像的指定区域中。
    
    参数:
    original_image (torch.Tensor): 原始图像
    generated_image (torch.Tensor): 生成的图像
    mask (torch.Tensor): 掩码图像, 1表示需要inpaint的区域, 0表示保留原图
    
    返回:
    torch.Tensor: 输出的合成图像
    """
    # 将三个输入tensor转换到相同的设备上
    device = original_image.device
    mask = mask.to(device)
    generated_image = generated_image.to(device)
    
    # 使用掩码融合原图和生成的图像
    output_image = original_image.clone()
    # print(output_image.shape, mask.shape, generated_image.shape)
    # print(output_image.shape, mask.shape, generated_image.shape)
    reference_image = output_image * mask
    output_image = original_image * (1 - mask) + generated_image * mask
    generated_image = generated_image * mask
    
    return output_image, generated_image, reference_image
# images_predict_slice, generated_image, reference_image = inpaint_image(images[:,:,:,:], d_images[:,:,:,:], masks[:,:,:,:])


import argparse
import re
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
args.run_name = "DDPM_conditional"
args.batch_size = 2
args.image_size = 96
args.dataset_path =  "/public/home/hansy2022/taotl/DDPM_brain/test_data/test_data"
args.generated_data_path = "/public/home/hansy2022/taotl/DDPM_brain/test_data/generated_data"
args.device = "cuda"
args.lr = 1e-4
args.train = True
args.shuffle = False
device = 'cuda'
# dataloader = get_data_inference(args)
# model = UNet_conditional_concat_Large().to(device)
model = UNet_conditional_concat_with_mask().to(device)
ckpt = torch.load("./models/DDPM_conditional/38_ema_ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=args.image_size, device=device)


test_dataloader = get_data_inference(args)
print(len(test_dataloader))
pbar_test = tqdm(test_dataloader)
for i, (images, cropped_images, masks, path) in enumerate(pbar_test):
    modefied_images = cropped_images
    b, _, _, _ = cropped_images.shape
    # print(b)
    # print(image_without_healthy.shape)
    masks = masks.to(torch.float)
    # print(masks.shape)
    modefied_images = modefied_images.to(device)
    d_images = diffusion.sample(model, n=b, labels=modefied_images, masks=masks)
    images_predict_slice, generated_image, reference_image = inpaint_image(cropped_images[:,:,:,:], d_images[:,:,:,:], masks[:,:,:,:])
    img = cropped_images.cpu()[:,0,:,:]
    d_images_new = images_predict_slice.cpu()
    dd_img = d_images.cpu()[:, 0, :, :]
    ref_images_new = reference_image.cpu()



    for index in range(b):
        dd_img_new = dd_img[index, :, :]
        d_img = d_images_new[index, 0, :, :]
        ref_img = ref_images_new[index, 0, :, :]
        orgin_path = path[index]
        
        filename = os.path.split(orgin_path)[-1]
        # print(orgin_path, filename)
        match = re.search(r'96_slice_(\d+)\.npz', filename)
        number = match.group(1)
        # print(z, number)
        subject_name = os.path.basename(os.path.dirname(orgin_path))
        generated_img_save_folder = os.path.join(args.generated_data_path, subject_name)
        os.makedirs(os.path.join(args.generated_data_path, subject_name), exist_ok=True)
        generated_img_save_path = os.path.join(generated_img_save_folder, f'generated_slice_{number}.npz')
        # # print(generated_img_save_path)
        np.savez(generated_img_save_path,
                 image = d_img)
