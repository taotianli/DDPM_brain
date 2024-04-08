
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
"""
需要修改的地方：
1:把label换成image encoder
2:image inpainting 需要把生成的部分和原图拼接,读取的是原图和mask
3:重写dataloader,修改get_args
4:按照切片的方式读取数据
5:改进guidance的方式(corss attention)
6:最后拼接成一个3D图像
7:加上实际的评价指标
8:留出一部分数据做测试用
9:把尺寸减少，同时每个subject只计算mask相关的slice
10:看一下batch size的问题
"""

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
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

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
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
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
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

# 使用示例
# original_image = torch.randn(1, 3, 256, 256)  # 原始图像
# generated_image = torch.randn(1, 3, 256, 256)  # 生成的图像
# mask = torch.randint(0, 2, (1, 1, 256, 256), dtype=torch.float32)  # 掩码图像

# output_image = inpaint_image(original_image, generated_image, mask)
# print(output_image.shape)  # 输出合成后的图像尺寸

def train_each_subject(images, device, diffusion, model, mse, optimizer, ema, pbar, ema_model, labels, masks):
    print(images.shape)
    b, h, w, len = images.shape
    # labels = mask

    images_predict = torch.zeros_like(labels)
    noise_predict = torch.zeros_like(labels)
    loss = 0
    for slice in range(50):
        print(slice)
        images_slice = images[:,:,:,slice]
        labels_slice = labels[:,:,:,slice]
        #去掉一个维度
        images_slice = images_slice.unsqueeze(0)
        labels_slice = labels_slice.unsqueeze(0)
        images_slice = images_slice.to(device)
        labels_slice = labels_slice.to(device)
        images_slice = images_slice.to(torch.float)
        labels_slice = labels_slice.to(torch.float)


        t = diffusion.sample_timesteps(images_slice.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images_slice, t)
        # if np.random.random() < 0.1: #这两行是做什么的
        #     labels = None
        predicted_noise = model(x_t, t, labels_slice)
        images_predict[:,:,:,slice] = predicted_noise
        noise_predict[:,:,:,slice] = noise
        images_predict_slice = inpaint_image(images[:,:,:,slice], images_predict[:,:,:,slice], masks[:,:,:,slice])
        noise_predict_slice = inpaint_image(images[:,:,:,slice], noise_predict[:,:,:,slice], masks[:,:,:,slice])
        loss = loss + mse(noise_predict_slice, images_predict_slice)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return noise_predict, images_predict, loss


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, cropped_images, masks, _) in enumerate(pbar):
            images = images.to(device)
            labels = cropped_images.to(device)
            # t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # x_t, noise = diffusion.noise_images(images, t)
            # if np.random.random() < 0.1:
            #     labels = None
            # predicted_noise = model(x_t, t, labels)
            # loss = mse(noise, predicted_noise)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # ema.step_ema(ema_model, model)

            # pbar.set_postfix(MSE=loss.item())
            noise_predict, images_predict, loss = train_each_subject(images, device, diffusion, model, mse, optimizer, ema, pbar, ema_model, labels, masks)
            # loss = mse(noise_predict, images_predict)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # if epoch % 10 == 0:
        #     labels = torch.arange(10).long().to(device)
        #     sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        #     ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
        #     # plot_images(sampled_images)
        #     save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        #     save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
        #     torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        #     torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
        #     torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 300
    args.batch_size = 1
    args.image_size = 64
    args.dataset_path =  r"D:\BraTS\ASNR_test"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

