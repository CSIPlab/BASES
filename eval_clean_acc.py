"""
Evaluate the accuracy of models on clean images 

To make sure that the classifiers are working properly

"""

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils_bases import load_imagenet_1000, get_logits_probs, load_model


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


device = 'cuda:0'
im_root = 'imagenet1000'
img_paths, gt_labels, tgt_labels = load_imagenet_1000(im_root)


surrogate_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']

victim_names = ['vgg19', 'densenet121', 'resnext50_32x4d']


for model_name in surrogate_names + victim_names:
    model = load_model(model_name, device)

    pred_labels = []
    for im_idx in tqdm(range(len(img_paths))):
        im = Image.open(img_paths[im_idx]).convert('RGB')
        im = np.array(im).astype(np.float32) / 255
        gt_label = gt_labels[im_idx]
        
        logits, _ = get_logits_probs(im, model)
        pred_label = logits.argmax()
        # print(pred_label)
        # print(pred_label == gt_label)

        pred_labels.append(pred_label)
    accuracy = np.mean(np.array(pred_labels) == np.array(gt_labels))
    print(model_name, accuracy)


# plot the fig of accuracy
# acc = [0.902, 0.85, 0.688, 0.873, 0.893, 0.945, 0.945, 0.894, 0.949, 0.968, 0.832, 0.932, 0.936, 0.799, 0.829, 0.808, 0.944, 0.956, 0.869, 0.939, 0.891, 0.917, 0.939]
# print(np.mean(acc), np.std(acc))
# print(np.max(acc), np.min(acc))