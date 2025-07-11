<div align="center">

# Event-Customized Image Generation

Zhen Wang<sup>1</sup>, Yilei Jiang<sup>1</sup>, Dong Zheng<sup>1</sup>, Jun Xiao<sup>1</sup>, [Long Chen](https://zjuchenlong.github.io/)<sup>2</sup>*  

<sup>1</sup>Zhejiang University  
<sup>2</sup>The Hong Kong University of Science and Technology (HKUST)  

**ICML 2025**


<a href="https://arxiv.org/abs/2410.02483">
<img src='https://img.shields.io/badge/arxiv-FreeEvent-blue' alt='Paper PDF'></a>



</div>

## 📖 Abstract


<div align="center">
  <img src="docs/intro.png"  width="100%">
</div>



Customize your event (actions, poses, relations, interactions) with one single image with our FreeEvent !




## 🚀 Quick Start


### 1. Setup

Create the environment and install the dependencies by running:

```
conda create -n free-event python=3.9
conda activate free-event
pip install -r requirements.txt
```


### 2. Data Preparation

Prepare your input data following this structure in the `input/` folder:

```
input/
└── your_event_name/
    ├── img.jpg                # Reference image (512×512 recommended)
    ├── mask0.png              # Entity mask 0 (binary mask)
    ├── mask1.png              # Entity mask 1 (optional)
    └── config.yaml            # Configuration file
```


Configuration File (config.yaml) Guide:
```
# Required parameters
entity: 2                                 # Number of entities
prompt: "a Jedi, a lightsaber"            # Target entity


# Entity attention control
entity_mask_to_tokens: 
- - 2                                # Token ind for mask0
- - 5                                # Token ind for mask1
  
entity_token_weights:
- - 1                                  # Weight for mask0 token
- - 1                                  # Weight for mask1 token

# Hyperparameters (default values shown)
guidance_scale: 15
n_timesteps: 50
fe_f_t: 1                                 # Feature injection ratio
fe_attn_t: 0.8                            # Attention injection ratio
fe_cross_attn_reg_t: 1                    # Attention regulation ratio
fe_cross_attn_guid_t: 0.2                 # Attention guidance ratio
```

See complete examples in `input/event2/config.yaml`.


### 3. Running FreeEvent

Run the following command for event customization:

```
python run.py --config_path <config_path>
```

We have prepared several examples, such as:

```
python run.py --config_path input/event2/config.yaml
python run.py --config_path input/event8/config.yaml
```

Have fun!

## 🌄 Combination of Event-Subject Customization

FreeEvent can aslo be easily extend to Event-Subject Customization.

<div align="center">
  <img src="docs/combine.png"  width="100%">
</div>


## 🗓️ TODO
- [x] Release initial code  
- [ ] Release benchmark  
- [ ] Release demo for Event-Subject Customization


## 🖊️ BibTeX
If you find this project useful in your research, please consider cite:

```bibtex
@article{wang2024event,
  title={Event-Customized Image Generation},
  author={Wang, Zhen and Jiang, Yilei and Zheng, Dong and Xiao, Jun and Chen, Long},
  journal={arXiv preprint arXiv:2410.02483},
  year={2024}
}
```

## 🙏 Acknowledgements
We thank to [Stable Diffusion](https://github.com/CompVis/stable-diffusion),  [PnP-Diffusers](https://github.com/MichalGeyer/pnp-diffusers), [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt), [Layout-Guidance](https://github.com/silent-chen/layout-guidance)
