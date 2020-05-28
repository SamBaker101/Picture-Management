'''
Sam Baker - 05/15/2020
deer picture source: personal
jazz picture source: https://www.flickr.com/photos/29713047@N00/157845230
'''

from PIL import Image
import torch
import torchvision
from Helper import Picture
from Helper import Filters
import matplotlib.pyplot as plt

def main():

    load_path = 'jazz.jpg'
    save_path = 'jazz_test.jpg'

    image = Picture(load_path)
    Filter = Filters()

    print('Width: ', image.width)
    print('Height', image.height)

    image.cropImage(70, 50, 300, 200)

    image.maxPool(kernel_size=4, stride=2)
    filter = Filter.edge33()
    image.conv1(filter, stride=2)
    image.relu1()
    filter = Filter.edge33()
    image.conv1(filter, stride=1)
    image.relu1()

    print(image.tensor)



    image.showImage()
    image.plotTensor(0)

    image.saveImage(save_path)


main()
