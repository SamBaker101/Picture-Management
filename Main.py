'''
Sam Baker - 05/15/2020
deer picture source: personal
jazz picture source: https://www.flickr.com/photos/29713047@N00/157845230
'''

from PIL import Image
import torch
import torchvision
from Helper import Picture
import matplotlib.pyplot as plt

def main():

    load_path = 'jazz.jpg'
    save_path = 'jazz_test.jpg'

    image = Picture(load_path)

    print('Width: ', image.width)
    print('Height', image.height)

    image.cropImage(70, 50, 300, 200)

    filter1 = torch.tensor([[-1., -1., 1., -1., -1.],
                            [-1., 1., 0, 1., -1.],
                            [1., 0, 0, 0, 1.],
                            [-1., 1., 0, 1., -1.],
                            [-1., -1., 1., -1., -1.]])

    image.halfResLow()
    image.mmFilter(filter1)
    image.relu1()
    image.maxPool(5, 1)
    image.mmFilter(filter1)
    image.maxPool(5, 1)
    image.showImage()
    image.plotTensor(0)
    image.saveImage(save_path)


main()
