import torch.utils.data as data
from PIL import Image
import os

def default_loader(path):
    img = Image.open(path).convert('RGB')
    return img

class CreateDataset(data.Dataset):
    def __init__(self, folder, transform=None, target_transform=None, loader=default_loader):
        classes = os.listdir(folder)
        imgs = []
        for c in classes:
            label = classes.index(c)
            images = os.listdir('{}/{}/images/'.format(folder, c))
            for image in images:
                imgs.append(('{}/{}/images/{}'.format(folder, c, image), label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.classes = classes

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)