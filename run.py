import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from pathlib import Path
from typing import List, Optional

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMScheduler, StableDiffusionPipeline, LMSDiscreteScheduler

from utils import *

# Suppress partial model loading warning
torch.autograd.set_detect_anomaly(True)
logging.set_verbosity_error()


class FreeEvent(nn.Module):
    def __init__(self, config):
        """Initialize FreeEvent with given configuration.
        
        Args:
            config (dict): Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        self.device = config["device"]
        sd_version = config["sd_version"]

        # Validate and set stable diffusion version
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, 
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        # Set up model components
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.generator = torch.Generator("cuda").manual_seed(1)

        # Set up schedulers
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)

        self.noise_scheduler = LMSDiscreteScheduler(
            beta_start=0.00085, 
            beta_end=0.012,
            beta_schedule="scaled_linear", 
            num_train_timesteps=1000
        )
        print('SD model loaded')

        # Set up event and entity parameters
        self.event = config["event"]
        self.entity_num = int(config["entity"])
        self.entity_mask_to_tokens = config["entity_mask_to_tokens"]
        self.entity_token_weights = config["entity_token_weights"]
        self.controller = AttentionStore()

        # Load data and embeddings
        self.image, self.eps = self.get_data()
        self.text_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])
        self.guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

        self.masks = self.load_masks()
        self.loss = torch.tensor(10000)

    def load_masks(self):
        """Load mask images for entities.
        
        Returns:
            list: List of mask tensors
        """
        masks = []
        for i in range(self.entity_num):
            mask_path = f'input/{self.event}/mask{i}.png'
            mask_image = Image.open(mask_path).convert("L")
            array = np.array(mask_image, dtype=np.float32) / 255.0
            mask_tensor = torch.tensor(array)
            mask_tensor = (mask_tensor > 0.5).float()
            masks.append(mask_tensor)
        return masks
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        """Get text embeddings for given prompts.
        """
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Get unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Concatenate for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        """Decode latent representation to image.
        
        Args:
            latent (torch.Tensor): Latent representation
            
        Returns:
            torch.Tensor: Decoded image tensor
        """
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode images to latent representation.
        
        Args:
            imgs (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Encoded latent representation
        """
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        """Load and process input data.
        
        Returns:
            tuple: (image tensor, noisy latent tensor)
        """
        # Load image
        img_path = f'input/{self.event}/img.jpg'
        image = Image.open(img_path).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)

        # Get noise
        noisy_latent = torch.randn((1, 4, 64, 64), device=self.device)

        # Get latent of reference image
        image_pil = T.Resize(512)(Image.open(img_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
        self.ref_latent = self.encode_imgs(image)

        return image, noisy_latent

    def denoise_step(self, x, t, index):
        """Perform a single denoising step.
        
        Args:
            x (torch.Tensor): Current latent representation
            t (torch.Tensor): Current timestep
            index (int): Current step index
            
        Returns:
            torch.Tensor: Denoised latent representation
        """
        # Register the time step and features in event transferring path
        # Register the time step and attention control in entity switching path
        register_time(self, t.item())
        register_attention_control(self, self.controller, t.item())

        # 1. Entity switching path
        iteration = 0
        while iteration < 5 and index < self.cross_guidance_steps:
            x = x.requires_grad_(True)
            latent_model_input = x
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=self.text_embeds.chunk(2)[1]
            )['sample']

            agg_attn = aggregate_attention(
                self.controller,
                res=64,
                from_where=("up", "down"),
                is_cross=True
            )
            attn_loss = 0
            
            for agg_attn_per in agg_attn:
                size_att = int((agg_attn_per.shape[0]))
                token_num = 0
                for mask_id in range(len(self.masks)):
                    # Get the corresponding target tokens and weights of each reference mask
                    token_inds = self.entity_mask_to_tokens[mask_id]
                    weights = self.entity_token_weights[mask_id]

                    gt_mask = self.masks[mask_id]
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(0).unsqueeze(0), 
                        size=(size_att, size_att)
                    ).squeeze(0).squeeze(0).to(agg_attn_per.device)

                    # Each target entity may contain multiple tokens (e.g. polar bear)
                    for j in range(len(token_inds)):
                        token = token_inds[j]
                        weight = weights[j]
                        token_num += 1
                        asset_attn_mask1 = agg_attn_per[..., token]
                        activation_value1 = (asset_attn_mask1 * gt_mask).reshape(1, -1).sum(dim=-1) / \
                                           asset_attn_mask1.reshape(1, -1).sum(dim=-1)
                        attn_loss += torch.mean((1 - activation_value1) ** 2) * weight

            # Average the attention loss
            self.loss = attn_loss / (token_num * len(agg_attn))
            grad_cond = torch.autograd.grad(self.loss.requires_grad_(True), [x])[0]

            # Cross attention guidance
            x = x - grad_cond * self.noise_scheduler.sigmas[index] ** 2
            iteration += 1
            torch.cuda.empty_cache()

        # 2. Event transferring path
        with torch.no_grad():
            # Perform forward process on the latent of reference image
            noise = randn_tensor(
                self.ref_latent.shape,
                generator=self.generator,
                device=self.ref_latent.device,
                dtype=self.ref_latent.dtype
            )
            source_latents = self.scheduler.add_noise(
                self.ref_latent,
                noise,
                t.reshape(1,)
            )
            
            # Get the input latent
            latent_model_input = torch.cat([source_latents] + ([x] * 2))

            # Compute text embeddings
            text_embed_input = torch.cat([self.guidance_embeds, self.text_embeds], dim=0)

            # Apply the denoising network
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embed_input
            )['sample']

            # Perform classifier-free guidance
            _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
            noise_pred = noise_pred_uncond + self.config["guidance_scale"] * \
                        (noise_pred_cond - noise_pred_uncond)

            # Compute the denoising step with the model
            denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
            torch.cuda.empty_cache()

        return denoised_latent
    
    def init_fe(self, conv_injection_t, qk_injection_t, cross_att_regulation_t, cross_att_guidance_t):
        """Initialize parameters.
        
        Args:
            conv_injection_t (int): Timestep for conv feature injection
            qk_injection_t (int): Timestep for self-attention map injection
            cross_att_regulation_t (int): Timestep for cross attention regulation
            cross_att_guidance_t (int): Timestep for cross attention guidance
        """
        # Timesteps for event transferring path
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []

        # Timesteps for entity switching path
        self.cross_regulation_timesteps = self.scheduler.timesteps[:cross_att_regulation_t] if cross_att_regulation_t >= 0 else []
        self.cross_guidance_steps = cross_att_guidance_t

        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_fe(self, seed):
        """Run event customization process.
        """
        fe_f_t = int(self.config["n_timesteps"] * self.config["fe_f_t"])
        fe_attn_t = int(self.config["n_timesteps"] * self.config["fe_attn_t"])
        fe_cross_att_reg_t = int(self.config["n_timesteps"] * self.config["fe_cross_attn_reg_t"])
        fe_cross_att_guid_t = int(self.config["n_timesteps"] * self.config["fe_cross_attn_guid_t"])
        
        self.init_fe(
            conv_injection_t=fe_f_t,
            qk_injection_t=fe_attn_t,
            cross_att_regulation_t=fe_cross_att_reg_t,
            cross_att_guidance_t=fe_cross_att_guid_t
        )
        customized_img = self.sample_loop(self.eps, seed)

    def sample_loop(self, x, seed):
        """Run the sampling loop to generate image.
        
        Returns:
            torch.Tensor: Generated image
        """
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t, i)

            decoded_latent = self.decode_latent(x)

            os.makedirs(f'{self.config["output_path"]}/{self.event}', exist_ok=True)
            T.ToPILImage()(decoded_latent[0]).save(
                f'{self.config["output_path"]}/{self.event}/img.jpg'
            )
                
        return decoded_latent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='input/event2/config.yaml')
    opt = parser.parse_args()
    
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config["seed"])
    print(config)
    fe = FreeEvent(config)
    fe.run_fe(config["seed"])
