import pickle
import numpy as np
import math
import tensorflow as tf
import cv2
import os
import os.path as osp
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def representative_dataset_gen():
    size = 100
    for i in range(size):
        lr_path = 'data/DIV2K/bin/DIV2K_train_LR_bicubic/X3/{:04d}x3.pt'.format(i+1)
        print('representative data: [{}]/[{}]'.format(i, size))
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)
        lr = lr.astype(np.float32)
        lr = np.expand_dims(lr, 0)
        yield [lr]

# set input tensor to [1, 360, 640, 3] for testing time
def representative_dataset_gen_time():
    size = 1
    for i in range(size):
        lr_path = 'data/DIV2K/bin/DIV2K_train_LR_bicubic/X3/{:04d}x3.pt'.format(i+1)
        print('representative data: [{}]/[{}]'.format(i, size))
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)
        lr = lr.astype(np.float32)
        lr = np.expand_dims(lr, 0)
        if lr.shape[1] >=360 and lr.shape[2] >= 640:
            yield [lr[:, 0:360, 0:640, :]]
        else:      
            continue


def quantize(model_path, quantized_model_path, time=False):
    if time:
        base, ext = osp.splitext(quantized_model_path)
        quantized_model_path = base + '_time' + ext
        tensor_shape = [1, 360, 640, 3]
        rep = representative_dataset_gen_time
    else:
        tensor_shape = [1, None, None, 3]
        rep = representative_dataset_gen

    model = tf.saved_model.load(model_path)#, custom_objects={'tf': tf})

    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(tensor_shape)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter=True
    converter.experimental_new_quantizer=True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quantized_tflite_model = converter.convert()
    open(quantized_model_path, 'wb').write(quantized_tflite_model)


def evaluate(quantized_model_path, save_path):

    interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    IS, IZ = input_details[0]['quantization']
    OS, OZ = output_details[0]['quantization']
    print('Input Scale: {}, Zero Point: {}'.format(IS, IZ))
    print('Output Scale: {}, Zero Point: {}'.format(OS, OZ))
    psnr = 0.0
    for i in range(801, 901):
        lr_path = 'data/DIV2K/bin/DIV2K_train_LR_bicubic/X3/0{}x3.pt'.format(i)
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)
        h, w, c = lr.shape
        lr = np.expand_dims(lr, 0).astype(np.float32)
        #lr = np.round(lr/IS+IZ).astype(np.uint8)
        lr = lr.astype(np.uint8)
        hr_path = 'data/DIV2K/bin/DIV2K_train_HR/0{}.pt'.format(i)
        with open(hr_path, 'rb') as f:
            hr = pickle.load(f)
        hr = np.expand_dims(hr, 0).astype(np.float32)
        interpreter.resize_tensor_input(input_details[0]['index'], lr.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], lr)
        interpreter.invoke()

        sr = interpreter.get_tensor(output_details[0]['index'])
        #sr = np.clip(np.round((sr.astype(np.float32)-OZ)*OS), 0, 255)
        sr = np.clip(sr, 0, 255)
        b, h, w, c = sr.shape
        # save image
        save_name = osp.join(save_path, '{:04d}x3.png'.format(i))
        cv2.imwrite(save_name, cv2.cvtColor(sr.squeeze().astype(np.uint8), cv2.COLOR_RGB2BGR))

        mse = np.mean((sr[:, 1:h-1, 1:w-1, :].astype(np.float32) - hr[:, 1:h-1, 1:w-1, :].astype(np.float32)) ** 2)
        singlepsnr =  20. * math.log10(255. / math.sqrt(mse))
        print('[{}]/[100]: {}'.format(i, singlepsnr))
        psnr += singlepsnr
    print(psnr / 100)


if __name__ == '__main__':
    #name = 'base7_D4C28_bs16ps64_lr1e-3'
    name = 'base7_D4C28_bs16ps64_lr1e-3_qat'
    model_path = 'experiment/{}/best_status'.format(name)
    save_path = 'experiment/{}/visual'.format(name)
    quantized_model_path = 'TFMODEL/{}.tflite'.format(name)

    quantize(model_path, quantized_model_path, time=False)
    quantize(model_path, quantized_model_path, time=True)
    
    evaluate(quantized_model_path, save_path)
