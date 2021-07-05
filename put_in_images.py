import os

folder = 'dataset/tiny-imagenet-200/val/'

classlist = os.listdir(folder)

for class in classlist:
    imagelist = os.listdir(folder + class)
    os.makedirs(folder + class + '/images')
    for image in imagelist:
        os.rename(folder + class + '/' + image, folder + class + '/images/' + image)