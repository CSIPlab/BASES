import torch
import torchvision.transforms as transforms

from random import randint

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


class Interpolate(torch.nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class Normalise(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i])/self.std[i]
        return x


# Only intended for use on untargeted attacks.
def margin_loss(logits, y):
    class_org = y.item()
    num_classes = logits.size(1)
    cover_orig_logit = torch.zeros(1, num_classes).to(logits.device)
    cover_orig_logit[0, y] = float('inf')
    class_tgt = (logits - cover_orig_logit).argmax(1, keepdim=True).item()
    logit_org = logits[0, class_org]
    logit_target = logits[0, class_tgt]
    loss = -logit_org + logit_target
    return loss, class_org, class_tgt


def generate_data_transform(transform_type):
    if transform_type == 'imagenet_common_224':
        image_width = 224
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_width),
            transforms.ToTensor()])
    elif transform_type == 'imagenet_inception_299':
        image_width = 299
        data_transform = transforms.Compose([
            transforms.Resize(342),
            transforms.CenterCrop(image_width),
            transforms.ToTensor()])
    else:
        raise ValueError("You've specified an invalid data transform.")
    return data_transform, image_width


def any_imagenet_id_but(avoid_id):
    random_class = randint(0, 999)
    return any_imagenet_id_but(avoid_id) if random_class == id else random_class
