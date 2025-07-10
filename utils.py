"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc
import random
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, List, Dict, Optional
import torch.nn.functional as F
from collections import defaultdict
from diffusers.models.cross_attention import CrossAttention


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class FECrossAttnProcessor:
    def __init__(self, controller, place_in_unet, qk_t, cross_t, name, masks, entity_mask_to_tokens):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.qk_t = qk_t
        self.cross_t = cross_t
        self.name = name
        self.masks = masks
        self.entity_mask_to_tokens = entity_mask_to_tokens

    def attention_Regularization(self, attention_probs):
        size = int(np.sqrt(attention_probs.shape[-2]))

        for i in range(len(self.masks)):
            word_ind = self.entity_mask_to_tokens[i]
            for word in word_ind:
                attention_probs[:, :, word] = attention_probs[:, :, word] * F.interpolate(
                    self.masks[i].unsqueeze(0).unsqueeze(0), 
                    (size, size)
                ).flatten(0).unsqueeze(0).to(attention_probs.device)
        return attention_probs

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        to_out = attn.to_out
        if isinstance(to_out, torch.nn.modules.container.ModuleList):
            to_out = attn.to_out[0]
        else:
            to_out = attn.to_out

        batch_size, sequence_length, _ = hidden_states.shape

        if batch_size == 3:
            query = attn.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )

            # Event transferring path: self-attention map injection
            if (not is_cross and self.qk_t is not None and 
                (self.t in self.qk_t or self.t == 1000) and 
                self.name.startswith('up_blocks') and 
                self.name.endswith('attn1.processor')):

                if (int(self.name[len("up_blocks.")]) == 1 and 
                    int(self.name[len("up_blocks.1.attentions.")]) == 0):
                    key = attn.to_k(encoder_hidden_states)
                    query = attn.head_to_batch_dim(query)
                    key = attn.head_to_batch_dim(key)
                else:
                    key = attn.to_k(encoder_hidden_states)
                    size = int(np.sqrt(key.shape[-2]))
                    mask_att = F.interpolate(
                        self.masks[0].unsqueeze(0).unsqueeze(0), 
                        (size, size)
                    ).flatten(0).unsqueeze(0).to(key.device)
                    mask_att = mask_att.view(-1)

                    source_batch_size = int(query.shape[0] // 3)  # 1

                    # Inject conditional
                    query[source_batch_size:2 * source_batch_size] = query[:source_batch_size]
                    key[source_batch_size:2 * source_batch_size] = key[:source_batch_size]
                    query[2 * source_batch_size:] = query[:source_batch_size]
                    key[2 * source_batch_size:] = key[:source_batch_size]

                    query = attn.head_to_batch_dim(query)
                    key = attn.head_to_batch_dim(key)
            else:
                key = attn.to_k(encoder_hidden_states)
                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)

            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * attn.scale

            h = attn.heads
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            attention_probs = sim.softmax(dim=-1)

            # Entity switching path: cross attention regulation
            if (is_cross and self.cross_t is not None and 
                (self.t in self.cross_t or self.t == 1000) and 
                (not self.name.startswith('mid_block'))):
                attention_probs = self.attention_Regularization(attention_probs)

            hidden_states = torch.einsum("b i j, b j d -> b i d", attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = to_out(hidden_states)
        else:
            query = attn.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )

            key = attn.to_k(encoder_hidden_states)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)

            value = attn.to_v(encoder_hidden_states)
            value = attn.head_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * attn.scale

            h = attn.heads
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            attention_probs = sim.softmax(dim=-1)
            self.controller(sim, is_cross, self.place_in_unet)

            hidden_states = torch.einsum("b i j, b j d -> b i d", attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = to_out(hidden_states)

        return hidden_states


def register_attention_control(model, controller, t):
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else model.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            att_id = int(name[len("up_blocks.x.attentions.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            att_id = int(name[len("down_blocks.x.attentions.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue
        cross_att_count += 1

        attn_procs[name] = FECrossAttnProcessor(
            controller=controller, 
            place_in_unet=place_in_unet, 
            qk_t=model.qk_injection_timesteps, 
            cross_t=model.cross_regulation_timesteps,
            name=name, 
            masks=model.masks, 
            entity_mask_to_tokens=model.entity_mask_to_tokens
        )
        setattr(attn_procs[name], 't', t)

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


def get_average_attention(controller):
    average_attention = {
        key: [item / controller.cur_step for item in controller.attention_store[key]]
        for key in controller.attention_store
    }
    return average_attention


def aggregate_attention(controller, res: int, from_where: List[str], is_cross: bool) -> List[torch.Tensor]:
    out = []
    attention_maps = get_average_attention(controller)
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] <= num_pixels:
                size = int(np.sqrt(item.shape[-2]))
                cross_maps = item.reshape(1, -1, size, size, item.shape[-1])[0]
                cross_maps = cross_maps.softmax(dim=-1)
                out.append(cross_maps.sum(0) / cross_maps.shape[0])
    return out


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def register_time(model, t):
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
            conv_module = model.unet.up_blocks[res].resnets[block]
            setattr(conv_module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


# Conv feature injection for event transferring
def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor
            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)

            if (hidden_states.shape[0] == 3 and 
                self.injection_schedule is not None and 
                (self.t in self.injection_schedule or self.t == 1000)):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # Inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # Inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
            return output_tensor
        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
