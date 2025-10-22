# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 12:27:46 2025

@author: User
"""

#%%
import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import lightning as L


class DiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(sample_size=32)
        self.scheduler = diffusers.schedulers.DDPMScheduler()

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    

class DiffusionData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.augment = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def prepare_data(self):
        load_dataset("cifar10")

    def train_dataloader(self):
        dataset = load_dataset("cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=128, shuffle=True, num_workers=12)


if __name__ == "__main__":
    model = DiffusionModel()
    data = DiffusionData()
    trainer = L.Trainer(max_epochs=150, precision="bf16-mixed")
    trainer.fit(model, data)


#%% scale the model image size and model size
import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import lightning as L


class DiffusionModel(L.LightningModule):
    def __init__(self, image_scale=1, model_scale=1):
        super().__init__()
        # Image size mapping
        size_map = {1: 32, 2: 64, 3: 128}
        sample_size = size_map.get(image_scale, 32)

        # Model scaling: widen channels by factor
        base_channels = 128 * model_scale
        self.model = diffusers.models.UNet2DModel(
                        sample_size=sample_size,
                        in_channels=3,
                        out_channels=3,
                        layers_per_block=2,
                        block_out_channels=[
                            base_channels,
                            base_channels * 2,
                            base_channels * 4,
                            base_channels * 4,   # added fourth stage
                        ],
                    )
        self.scheduler = diffusers.schedulers.DDPMScheduler()

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    

class DiffusionData(L.LightningDataModule):
    def __init__(self, image_scale=1):
        super().__init__()
        size_map = {1: 32, 2: 64, 3: 128}
        image_size = size_map.get(image_scale, 32)

        self.augment = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def prepare_data(self):
        load_dataset("cifar10")

    def train_dataloader(self):
        dataset = load_dataset("cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=128, shuffle=True, num_workers=128)


if __name__ == "__main__":
    image_scale = 1  # 1=32, 2=64, 3=128
    model_scale = 1  # 1=default, 2=double, 3=triple
    model = DiffusionModel(image_scale=image_scale, model_scale=model_scale)
    data = DiffusionData(image_scale=image_scale)
    trainer = L.Trainer(max_epochs=150, precision="bf16-mixed")
    trainer.fit(model, data)