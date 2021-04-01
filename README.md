# Introduction
A good solution for [MAI2021 Competition(CVPR2021 Workshop)](https://competitions.codalab.org/competitions/28119). Our model can achieve the best trade-off between mobile inference time and reconstruction quality during validating phase, but unfortunately, following code is ignored during testing phase thus leading to incorrect dequantization. After correcting, we are confident enough that our method is the most efficient among existing methods on mobile devices such as Synaptics smart TV platform.
```bash
# In solvers/networks/base7.py
out = tf.keras.backend.clip(out, 0., 255.)
```

# Contribution of our model for INT8 Quantization Super-Resolution Mobile Network
* Anchor-based residual learning
For full-integer quantization which means all the weights and activations are int8, it's obvious a better choice to learn residual(always close to zero) rather than directly mapping low-resolution image to high-resolution image. In existing methods, residual learning can be divided into two categories: (1). Image space residual learning means passing the interpolated-input(bilinear, bicubic) to network output. (2).Feature space residual learning means passing the output of shallow convolutional layer to network output. For float32 quantized model, feature space residual learning is slightly better(+0.08dB). For int8 quantized model, image space residual learning is always better(+0.3dB) because it forces the whole network to learn subtle change, thus a set of continuous real-valued numbers can be represented more accurately using a fixed discrete set of numbers. However, bilinear resize and nearest neighbor resize is really slow on mobile device due to pixel-wise multiplication when doing coordinate mapping. Our anchor-based residual learning can enjoy the good property of image space residual learning while being as fast as feature space residual learning. The core operation is repeating input nine times(for x3 scale) and add it to the feature before depth-to-space. See our architecture in [model](https://github.com/NJU-Jet/SR_Mobile_Quantization/blob/master/solvers/networks/base7.py)

* Investigation of depth and width
Since cache memory is limited, some mechanisms such as feature fusion and attention mechanism are not suitable for mobile device. The network architecture design is limited, and it's always true that deeper or wider network leads to better performance, sacrificing inference speed at the same time. We use grid search to get the best trade-off state.

* Another more convolution after deep feature extraction
After deep feature extraction, existing methods use one convolution to map features to origin image space, followed by a depth-to-space(PixelShuffle in Pytorch) layer. We find that in image space, one more convolution can significantly improve the performance compared with adding one convolution in deep feature extraction stage(+0.11dB)

# Requirements
I also provide [requirements.yaml](https://github.com/NJU-Jet/SR\_Framework/blob/master/sr\_framework/requirements.yaml) for you to copy my conda environment. If you have anaconda, you can use the following codes:
1. First make sure your nvidia driver version is larger than 410.48
```bash
nvidia-smi
```
2. Create conda environment from yaml
```bash
conda env create -f requirements.yaml
source activate SR
```
If you want to figure out which file includes the libraries(take tensorboardX as an example), you can use:
```bash
grep -rn --color=auto 'tensorboardX'
```

# Train on DIV2K
1. Download DIV2K dataset from [EDVR](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md#REDS), unpack the tar file to any place you want.
2. Change ```dataroot_HR``` and ```dataroot_LR``` arguments in ```options/train/{model}.yaml``` to the place where DIV2K images are located.(change {model} according to your need)
3. Run(change {model} according to your need, --use_chop is for saving memory in validation stage):
```bash
python train.py --opt options/train/{model}.yaml --name {name} --scale 2 --lr 2e-4 --bs 16 --ps 64 --gpu_ids 0 --use_chop
```
You can also use the dafault setting which keeps the same with the original article by running:
```bash
python train.py --opt options/train/{model}.yaml --name {name} --scale {scale} --gpu_ids {ids}
```
**Note**: ```gpu_ids``` can be a series of numbers separated by comma, like ```0,1,3```.
The argument ```--name``` specifies the following save path:
* Log file will be saved in ```log/{name}.log```
* Checkpoint and current best weights will be saved in ```experiment/{name}/{epochs}/```
* Train/Val loss and psnr/ssim will be saved in ```experiment/{name}/records/```
* Visualization of Train and Validate will be saved in ```../Tensorboard/{name}/```


# Train on your own dataset
1. Change ```dataroot_HR``` and ```dataroot_LR``` arguments in ```options/train/{model}.yaml``` to the place where your images are located.(change {model} according to your need)
2. Change ```mode``` argument in ```options/train/{model}.yaml``` to ```TrainLRHR```
3. Run(change {model} according to your need, --use_chop is for saving memory in validation stage):
```bash
python train.py --opt options/train/{model}.yaml --name {model}_bs16ps64lr2e-4_x2 --scale 2 --lr 2e-4 --bs 16 --ps 64 --gpu_ids 0 --use_chop
```
* Log file will be saved in ```log/{name}.log```
* Checkpoint and current best weights will be saved in ```experiment/{name}/{epochs}/```
* Train/Val loss and psnr/ssim will be saved in ```experiment/{name}/records/```
* Visualization of Train and Validate will be saved in ```../Tensorboard/{name}/```

# Test on Benchmark(Set5, Set14, B100, Urban100, Mango109)
1. Download benchmark dataset from [EDVR](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md#REDS), unpack the tar file to any place you want.
2. Change ```dataroot_HR``` and ```dataroot_LR``` arguments in ```options/test/base.yaml``` to the place where benchmark images are located.
3. Run:
```bash
python test.py --opt options/test/base.yaml --dataset_name {dataset_name} --scale {scale} --which_model {model} --pretrained {pretrained_path}
```
For example:
```bash
python test.py --opt options/test/base.yaml --dataset_name Set5 --scale 2 --which_model EDSR --pretrained pretrained/EDSR.pth
```
* Psnr & ssim of each image will be printed on your screen.
