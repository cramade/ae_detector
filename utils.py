import os
import torch
import numpy as np

from torchvision.utils import save_image
from attack_pgd import projected_gradient_descent

def init_dir(base):
    if os.path.exists(base) != True:
        os.mkdir(base)

    if os.path.exists(base+'/test') != True:
        os.mkdir(base+'/test')
    for i in range (0, 10):
        if os.path.exists(f'{base}/test/{i}') != True:
            os.mkdir(f'{base}/test/{i}')

    if os.path.exists(base+'/train') != True:
        os.mkdir(base+'/train')
        for i in range (0, 10):
            if os.path.exists(f'{base}/train/{i}') != True:
                os.mkdir(f'{base}/train/{i}')

def attack_and_save_adv_imgs(model, images, base):
  total = len(images)
  for idx in range(0, total):
    image, label = images[idx]

    images = image.unsqueeze(0)
    labels = torch.tensor(label, dtype=torch.int8)
    adv_images = projected_gradient_descent(model, images, 0.3, 0.01, 40, np.inf)

    if (idx % 1000 == 0):
      print(f"{100*idx/total}% ({idx}/{total})")

    path = f"{base}/test/{labels.item()}/{idx}.png"
    if os.path.exists(path) != True:
      save_image(adv_images, path)
