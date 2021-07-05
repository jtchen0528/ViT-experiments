import os

folder = 'dataset/tiny-imagenet-200/val/'

classlist = os.listdir(folder)

for c in classlist:
    imagelist = os.listdir(folder + c)
    os.makedirs(folder + c + '/images')
    for image in imagelist:
        os.rename(folder + c + '/' + image, folder + c + '/images/' + image)