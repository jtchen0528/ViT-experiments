# %%

import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor
import torch.nn.functional as F
import torch.nn as nn
from transformers import ViTModel
import torchvision
from PIL import Image

# %%

transform = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor()])

train_ds = torchvision.datasets.ImageFolder(
    'tiny-imagenet-200/train/', transform=transform)
# %%
def default_loader(path):
    return Image.open(path).convert('RGB')

class EvalDataset(data.Dataset):
    def __init__(self, folder, transform=None, target_transform=None, loader=default_loader):
        fh = open(folder + 'val_annotations.txt', 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            label = list(train_ds.class_to_idx.keys()).index(words[1])
            imgs.append((folder + 'images/' + words[0], label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

eval_ds=EvalDataset(folder='tiny-imagenet-200/val/', transform=transform)
test_ds=EvalDataset(folder='tiny-imagenet-200/val/', transform=transform)
# %%


class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=3):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None


# %%
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5

# %%
# %%
# Define Model
model = ViTForImageClassification(len(train_ds.classes))
# Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained(
    'google/vit-base-patch16-224-in21k')
# Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Cross Entropy Loss
loss_func = nn.CrossEntropyLoss()
# Use GPU if available
device = 3

if torch.cuda.is_available():
    model.to(3)

# %%
torch.cuda.is_available()
# %%
print("Number of train samples: ", len(train_ds))
print("Number of test samples: ", len(test_ds))
print("Detected Classes are: ", train_ds.class_to_idx)

train_loader = data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = data.DataLoader(
    eval_ds, batch_size=1, shuffle=False)
# %%
# Train the model
for epoch in range(EPOCHS):
    for step, (x, y) in enumerate(train_loader):
        # Change input array into list with each batch being one element
        x = np.split(np.squeeze(np.array(x)), BATCH_SIZE)
        # Remove unecessary dimension
        for index, array in enumerate(x):
            x[index] = np.squeeze(array)
        # Apply feature extractor, stack back into 1 tensor and then convert to tensor
        x = torch.tensor(
            np.stack(feature_extractor(x)['pixel_values'], axis=0))
        # Send to GPU if available
        if torch.cuda.is_available():
            x, y = x.to(device), y.to(device)
        b_x = Variable(x)   # batch x (image)
        b_y = Variable(y)   # batch y (target)
        # Feed through model
        output, loss = model(b_x, None)
        print(output.shape, b_y.shape)
        # Calculate loss
        if loss is None:
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 50 == 0:
            # Get the next batch for testing purposes
            test = next(iter(test_loader))
            test_x = test[0]
            # Reshape and get feature matrices as needed
            test_x = np.split(np.squeeze(np.array(test_x)), BATCH_SIZE)
            for index, array in enumerate(test_x):
                test_x[index] = np.squeeze(array)
            test_x = torch.tensor(
                np.stack(feature_extractor(test_x)['pixel_values'], axis=0))
            # Send to appropirate computing device
            test_y = torch.tensor(test[1])
            if torch.cuda.is_available():
                test_x = test_x.to(device)
                test_y = test_y.to(device)
            # Get output (+ respective class) and compare to target
            test_output, loss = model(test_x, test_y)
            test_output = test_output.argmax(1)
            # Calculate Accuracy
            accuracy = (test_output == test_y).sum().item() / BATCH_SIZE
            print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)

torch.save(model.state_dict(), "checkpoints/vit_pretrained.pth")
# %%
