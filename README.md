# 1 Installation
## 1. Clone the repo
```bash
git clone --recursive https://github.com/thisiszy/Neural-Sim-NeRF.git
```

## 2. Virtual environment
Tested on Python/3.10.4 with gcc/8.2.0, cuda/11.7.0, nccl/2.11.4-1, cudnn/8.2.1.32

```bash
cd Neural-Sim-NeRF
python -m venv venv
source venv/bin/activate
./install.sh
```

# 2 Quick start
## 0. Generate training data
We use the [BlenderNerf](https://github.com/maximeraafat/BlenderNeRF) to generate train and test data.

Download our blender [scene file and hand model](https://drive.google.com/file/d/1LNgaFwF1b7Gk2J4h100xNmZyqre-7yyV/view?usp=share_link).

Use COS to generate 100 train and test pictures.

## 1. Train your own nerf
Download our [hand dataset](https://drive.google.com/file/d/1FSAwQpheviDZlX_RjlX8XA9HmPfwNW5E/view?usp=share_link) and extract to `data/` folder.

Train three one by one:
```bash
cd optimization
python train_nerf.py ../data/hand_palm
python train_nerf.py ../data/hand_fist
python train_nerf.py ../data/hand_yeah
```
You can also download our [pre-trained model](https://drive.google.com/file/d/1gYqc0Sf-ymqXjSuiapfkinT9RThVAwbr/view?usp=share_link).

If want to train your own dataset, please refer to [original ngp-torch repo](https://github.com/ashawkey/torch-ngp).

## 2. Train neural-sim

```bash
python neural_sim_main.py --config ../configs/nerf_param_ycbv_general.txt --object_id 1 --expname  exp_ycb_synthetic --ckpt PATH_TO_YOUR_MODEL(e.g hand_palm)
python neural_sim_main.py --config ../configs/nerf_param_ycbv_general.txt --object_id 2 --expname  exp_ycb_synthetic --ckpt PATH_TO_YOUR_MODEL(e.g hand_fist)
python neural_sim_main.py --config ../configs/nerf_param_ycbv_general.txt --object_id 8 --expname  exp_ycb_synthetic --ckpt PATH_TO_YOUR_MODEL(e.g hand_yeah)
```
