import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision.models as models # version 0.12
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


normalize = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# we use images with size 224
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

relu = torch.nn.ReLU()
softmax = torch.nn.Softmax(dim=1)
softmax_np = lambda x: np.exp(x) / np.sum(np.exp(x))
loss_ce_np = lambda x,y: -np.log(softmax_np(x)[y])  # x: list - logits, y: int - the label


model_names = [
    'alexnet', 
    'vgg11', 
    'vgg13', 
    'vgg16', 
    'vgg19', 
    'vgg11_bn', 
    'vgg13_bn', 
    'vgg16_bn', 
    'vgg19_bn',
    'resnet18', 
    'resnet34', 
    'resnet50', 
    'resnet101', 
    'resnet152', 
    'squeezenet1_0', 
    'squeezenet1_1',
    'densenet121', 
    'densenet161', 
    'densenet169', 
    'densenet201', 
    'inception_v3', 
    'googlenet', 
    'shufflenet_v2_x1_0', 
    'shufflenet_v2_x0_5',
    'mobilenet_v2', 
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'resnext50_32x4d', 
    'resnext101_32x8d', 
    'wide_resnet50_2', 
    'wide_resnet101_2', 
    'mnasnet1_0', 
    'mnasnet0_5',
    'efficientnet_b0',
    'efficientnet_b1',
    'efficientnet_b2',
    'efficientnet_b3',
    'efficientnet_b4',
    'efficientnet_b5',
    'efficientnet_b6',
    'efficientnet_b7',
    'regnet_y_400mf',
    'regnet_y_800mf',
    'regnet_y_1_6gf',
    'regnet_y_8gf',
    'regnet_x_400mf',
    'regnet_x_1_6gf',
    'regnet_x_8gf',
    'vit_b_16',
    'vit_b_32',
    'vit_l_16',
    'vit_l_32',
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    'convnext_large'
    ]


def load_model(model_name, device):
    """Load the model according to the idx in list model_names
    Args: 
        model_name (str): the name of model, chosen from the following list
        model_names = ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', \
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0', 'squeezenet1_1', \
            'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 'googlenet', 'shufflenet_v2_x1_0', 'shufflenet_v2_x0_5', \
            'mobilenet_v2', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet1_0', 'mnasnet0_5']
    Returns:
        model (torchvision.models): the loaded model
    """
    model = getattr(models, model_name)(pretrained=True).to(device).eval()
    return model


def get_logits_probs(im, model):
    """Get the logits of a model given an image as input
    Args:
        im (PIL.Image or np.ndarray): uint8 image (read from PNG file), for 0-1 float image
        model (torchvision.models): the model returned by function load_model
    Returns:
        logits (numpy.ndarray): direct output of the model
        probs (numpy.ndarray): logits after softmax function
    """
    device = next(model.parameters()).device
    im_tensor = preprocess(im).unsqueeze(0).to(device)
    logits = model(im_tensor)
    probs = softmax(logits)
    logits = logits.detach().cpu().numpy().squeeze() # convert torch tensor to numpy array
    probs = probs.detach().cpu().numpy().squeeze()
    return logits, probs


def get_label(im, model):
    """Get the label idx, ranging from 0-999
    Args:
        im (PIL.Image or np.ndarray): uint8 image (read from PNG file), for 0-1 float image
    """
    logits, probs = get_logits_probs(im, model)
    label_idx = logits.argmax().item()
    return label_idx


def get_loss(im, model, tgt_label, loss_name):
    """Get the loss
    Args:
        im (PIL.Image): uint8 image (read from PNG file)
        tgt_label (int): target label
    """
    loss_fn = get_loss_fn(loss_name, targeted=True)
    logits, _ = get_logits_probs(im, model)
    logits = torch.tensor(logits)
    tgt_label = torch.tensor(tgt_label)
    loss = loss_fn(logits, tgt_label)
    return loss


def load_imagenet_1000(dataset_root = "imagenet1000"):
    """
    Dataset downoaded form kaggle
    https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set
    Resized from 299x299 to 224x224

    Args:
        dataset_root (str): root folder of dataset
    Returns:
        img_paths (list of strs): the paths of images
        gt_labels (list of ints): the ground truth label of images 
        tgt_labels (list of ints): the target label of images 
    """
    dataset_root = Path(dataset_root)
    img_paths = list(sorted(dataset_root.glob('*.png')))
    gt_dict = defaultdict(int)
    tgt_dict = defaultdict(int)
    with open(dataset_root / "images.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gt_dict[row['ImageId']] = int(row['TrueLabel'])
            tgt_dict[row['ImageId']] = int(row['TargetClass'])
    gt_labels = [gt_dict[key] - 1 for key in sorted(gt_dict)] # zero indexed
    tgt_labels = [tgt_dict[key] - 1 for key in sorted(tgt_dict)] # zero indexed
    return img_paths, gt_labels, tgt_labels


def loss_cw(logits, tgt_label, margin=200, targeted=True):
    """c&w loss: targeted
    Args: 
        logits (Tensor): 
        tgt_label (Tensor): 
    """
    device = logits.device
    k = torch.tensor(margin).float().to(device)
    tgt_label = tgt_label.squeeze()
    logits = logits.squeeze()
    onehot_logits = torch.zeros_like(logits)
    onehot_logits[tgt_label] = logits[tgt_label]
    other_logits = logits - onehot_logits
    best_other_logit = torch.max(other_logits)
    tgt_label_logit = logits[tgt_label]
    if targeted:
        loss = torch.max(best_other_logit - tgt_label_logit, -k)
    else:
        loss = torch.max(tgt_label_logit - best_other_logit, -k)
    return loss


class CE_loss(nn.Module):

    def __init__(self, target=False):
        super(CE_loss, self).__init__()
        self.target = target
        self.loss_ce = torch.nn.CrossEntropyLoss()
        
    def forward(self, logits, label):
        loss = self.loss_ce(logits, label)
        if self.target:
            return loss
        else:
            return -loss


class CW_loss(nn.Module):

    def __init__(self, target=False):
        super(CW_loss, self).__init__()
        self.target = target
        self.loss_cw = loss_cw
        
    def forward(self, logits, label):
        return loss_cw(logits, label, targeted=self.target)
        

def get_loss_fn(loss_name, targeted=True):
    """get loss function by name
    Args: 
        loss_name (str): 'cw', 'ce', 'hinge' ...
    """
    if loss_name == 'ce':
        return CE_loss(targeted)
    elif loss_name == 'cw':
        return CW_loss(targeted)


def get_label_loss(im, model, tgt_label, loss_name, targeted=True):
    """Get the loss
    Args:
        im (PIL.Image): uint8 image (read from PNG file)
        tgt_label (int): target label
        loss_name (str): 'cw', 'ce'
    """
    loss_fn = get_loss_fn(loss_name, targeted=targeted)
    logits, _ = get_logits_probs(im, model)
    pred_label = logits.argmax()
    logits = torch.tensor(logits)
    tgt_label = torch.tensor(tgt_label)
    loss = loss_fn(logits, tgt_label)
    # get top 5 labels
    top5 = logits.argsort()[-5:]
    return pred_label, loss, top5


def get_adv(im, adv, target, w, pert_machine, bound, eps, n_iters, alpha, algo='pgd', fuse='loss', untargeted=False, intermediate=False, loss_name='ce'):
    """Get the adversarial image by attacking the perturbation machine
    Args:
        im (torch.Tensor): original image
        adv (torch.Tensor): initial value of adversarial image, can be a copy of im
        target (torch.Tensor): 0-999
        w (torch.Tensor): weights for models
        pert_machine (list): a list of models
        bound (str): choices=['linf','l2'], bound in linf or l2 norm ball
        eps, n_iters, alpha (float/int): perturbation budget, number of steps, step size
        algo (str): algorithm for generating perturbations
        fuse (str): methods to fuse ensemble methods. logit, prediction, loss
        untargeted (bool): if True, use untargeted attacks, target_idx is the true label.
        intermediate (bool): if True, save the perturbation at every 10 iters.
        loss_name (str): 'cw', 'ce', 'hinge' ...
    Returns:
        adv (torch.Tensor): adversarial output
    """
    # device = next(pert_machine[0].parameters()).device
    n_wb = len(pert_machine)
    if algo == 'mim':
        g = 0 # momentum
        mu = 1 # decay factor
    
    loss_fn = get_loss_fn(loss_name, targeted = not untargeted)
    losses = []
    if intermediate:
        adv_list = []
    for i in range(n_iters):
        adv.requires_grad=True
        input_tensor = normalize(adv/255)
        outputs = [model(input_tensor) for model in pert_machine]

        if fuse == 'loss':
            loss = sum([w[idx] * loss_fn(outputs[idx],target) for idx in range(n_wb)])
        elif fuse == 'prob':
            target_onehot = F.one_hot(target, 1000)
            prob_weighted = torch.sum(torch.cat([w[idx] * softmax(outputs[idx]) for idx in range(n_wb)], 0), dim=0, keepdim=True)
            loss = - torch.log(torch.sum(target_onehot*prob_weighted))
        elif fuse == 'logit':
            logits_weighted = sum([w[idx] * outputs[idx] for idx in range(n_wb)])
            loss = loss_fn(logits_weighted,target)

        losses.append(loss.item())
        loss.backward()
        
        with torch.no_grad():
            grad = adv.grad
            if algo == 'fgm':
                # needs a huge learning rate
                adv = adv - alpha * grad / torch.norm(grad, p=2)
            elif algo == 'pgd':
                adv = adv - alpha * torch.sign(grad)
            elif algo == 'mim':
                g = mu * g + grad / torch.norm(grad, p=1)
                adv = adv - alpha * torch.sign(g)

            if bound == 'linf':
                # project to linf ball
                adv = (im + (adv - im).clamp(min=-eps,max=eps)).clamp(0,255)
            else:
                # project to l2 ball
                pert = adv - im
                l2_norm = torch.norm(pert, p=2)
                if l2_norm > eps:
                    pert = pert / l2_norm * eps
                adv = (im + pert).clamp(0,255)

        if intermediate and i%10 == 9:
            adv_list.append(adv.detach())
    if intermediate:
        return adv_list, losses
    return adv, losses


def get_adv_np(im_np, target_idx, w_np, pert_machine, bound, eps, n_iters, alpha, algo='pgd', fuse='loss', untargeted=False, intermediate=False, loss_name='ce', adv_init=None):
    """Get the numpy adversarial image
    Args:
        im_np (numpy.ndarray): original image
        adv_init (numpy.ndarray): initialization of adv, if None, start with im_np
        target_idx (int): target label
        w_np (list of ints): weights for models
        untargeted (bool): if True, use untargeted attacks, target_idx is the true label.
    Returns:
        adv_np (numpy.ndarray): adversarial output
    """
    device = next(pert_machine[0].parameters()).device
    im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to(device)
    if adv_init is None:
        adv = torch.clone(im) # adversarial image
    else:
        adv = torch.from_numpy(adv_init).permute(2,0,1).unsqueeze(0).float().to(device)
    target = torch.LongTensor([target_idx]).to(device)
    w = torch.from_numpy(w_np).float().to(device)
    adv, losses = get_adv(im, adv, target, w, pert_machine, bound, eps, n_iters, alpha, algo=algo, fuse=fuse, untargeted=untargeted, intermediate=intermediate, loss_name=loss_name)
    if intermediate: # output a list of adversarial images
        adv_np = [adv_.squeeze().cpu().numpy().transpose(1, 2, 0) for adv_ in adv]
    else:
        adv_np = adv.squeeze().cpu().numpy().transpose(1, 2, 0)
    return adv_np, losses
    