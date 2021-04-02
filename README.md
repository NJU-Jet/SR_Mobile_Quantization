# Introduction
A good solution for [MAI2021 Competition(CVPR2021 Workshop)](https://competitions.codalab.org/competitions/28119). Our model can achieve the best trade-off between mobile inference time and reconstruction quality during validating phase, but unfortunately, following code is ignored during testing phase thus leading to incorrect dequantization. After correcting, we are confident enough that our method is the most efficient among existing methods on mobile devices such as Synaptics smart TV platform.
```python
# In solvers/networks/base7.py
out = tf.keras.backend.clip(out, 0., 255.)
```

# Contribution for INT8 Quantization SR Mobile Network
## Anchor-based residual learning

For **full-integer** quantization which means all the weights and activations are int8, it's obvious a better choice to learn residual(always close to zero) rather than directly mapping low-resolution image to high-resolution image. In existing methods, **residual learning** can be divided into two categories: (1). Image space residual learning means passing the interpolated-input(bilinear, bicubic) to network output. (2).Feature space residual learning means passing the output of shallow convolutional layer to network output. For float32 quantized model, feature space residual learning is slightly better(+0.08dB). For int8 quantized model, image space residual learning is always better(**+0.3dB**) because it forces the whole network to learn subtle change, thus a set of continuous real-valued numbers can be represented more accurately using a fixed discrete set of numbers. However, bilinear resize and nearest neighbor resize is really slow on mobile device due to pixel-wise multiplication when doing coordinate mapping. Our anchor-based residual learning can enjoy the good property of image space residual learning while being as fast as feature space residual learning. The core operation is repeating input nine times(for x3 scale) and add it to the feature before depth-to-space. See our architecture in [model](https://github.com/NJU-Jet/SR_Mobile_Quantization/blob/master/solvers/networks/base7.py).

## Investigation of depth and width

Since cache memory is limited, some mechanisms such as feature fusion, attention mechanism, residual block are not suitable for mobile device due to slow access to RAM. The network architecture design is limited, and it's always true that deeper or wider network leads to better performance, sacrificing inference speed at the same time. We use grid search to get the best trade-off state.

## Another more convolution after deep feature extraction

After deep feature extraction, existing methods use one convolution to map features to origin image space, followed by a depth-to-space(PixelShuffle in Pytorch) layer. We find that in image space, one more convolution can significantly improve the performance compared with adding one convolution in deep feature extraction stage(**+0.11dB**).

# Requirements
It should be noted that **tensorflow version** matters a lot because old versions don't include some layers such as depth-to-space, so you should make sure tf version is larger than 2.4.0. Another important thing is that only tf-nightly larger than 2.5.0 can perform arbitrary input shape quantization. I provide two conda environments, [tf.yaml](https://github.com/NJU-Jet/SR_Mobile_Quantization/blob/master/tf.yaml) for training and [tfnightly.yaml](https://github.com/NJU-Jet/SR_Mobile_Quantization/blob/master/tfnightly.yaml) for Post-Training Quantization(PTQ) and Quantization-Aware Training(QAT). You can use the following scripts to create two separate conda environments.
```bash
conda env create -f tf.yaml
conda env create -f tfnightly.yaml
```

# Pipeline
1. Train and validate on DIV2K. We can achieve **30.22dB** with **42.54K** parameters.
2. Post-Training Quantization: after int8 quantization, PSNR drops to **30.09dB**.
3. Quantization-Aware Training: Insert fake quantization nodes during training. PSNR increases to **30.15dB**, which means the model size becomes 4x smaller with only 0.07dB performance loss.

# Prepare DIV2K Data
Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and put DIV2K in data folder. Then the structure should look like:
> data
>> DIV2K
>>> DIV2K\_train\_HR
>>>> 0001.png

>>>> ...

>>>> 0900.png

>>> DIV2K\_train\_LR\_bicubic
>>>> X2
>>>>> 0001x2.png

>>>>> ...

>>>>> 0900x2.png

# Training
```bash
python train.py --opt options/train/base7.yaml --name base7_D4C28_bs16ps64_lr1e-3 --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0
```
**Note**: 
The argument ```--name``` specifies the following save path:
* Log file will be saved in ```log/{name}.log```
* Checkpoint and current best weights will be saved in ```experiment/{name}/best_status/```
* Visualization of Train and Validate will be saved in ```Tensorboard/{name}/```

You can use tensorboard to monitor the training and validating process by:
```bash
tensorboard --logdir Tensorboard
```

# Quantization-Aware Training
If you haven't worked with Tensorflow Lite and network quantization before, please refer to [official guideline](https://www.tensorflow.org/model_optimization/guide/quantization/training_example). This technology inserts fake quantization nodes to make the weights aware that themselves will be quantized. For this model, you can simply use the following script to perform QAT:
```bash
python train.py --opt options/train/base7_qat.yaml --name base7_D4C28_bs16ps64_lr1e-3_qat --scale 3  --bs 16 --ps 64 --lr 1e-3 --gpu_ids 0 --qat --qat_path experiment/base7_D4C28_bs16ps64_lr1e-3/best_status
```

# Convert to TFLite which can run on mobile device
``` bash
python generate_tflite.py
```
Then the converted tflite model will be saved in ```TFMODEL/```. ```TFMODEL/{name}.tflite``` is used for predicting high-resolution image(arbitary low-resolution input shape is allowed), while ```TFMODEL/{name}_time.tflite``` fixes model input shape to ```[1, 360, 640, 3]``` for getting inference time.

# Run TFLite Model on your own devices
1. Download AI Benchmark from the [Google Play](https://play.google.com/store/apps/details?id=org.benchmark.demo) / [website](https://polybox.ethz.ch/index.php/s/diruRfJZ4JqS4tZ) and run its standard tests.
2. After the end of the tests, enter the **PRO Model** and select the **Custom Model** tab there.
3. send your tflite model to your device and remember its location, then run the model.

# Contact
:) If you have any questions, feel free to contact 151220022@smail.nju.edu.cn
