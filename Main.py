'''
Sam Baker - 05/15/2020
jazz picture source: https://www.flickr.com/photos/29713047@N00/157845230
'''

from PIL import Image
import torch
import torchvision
import Helper
from Helper import Picture

def main():

    load_path = 'jazz.jpg'
    save_path = 'jazz_test.jpg'

    image = Picture(load_path)

    print('Width: ', image.width)
    print('Height', image.height)

    image.cropImage(120, 50, 200, 200)

    image.imageToTensor()
    print(image.tensor)

    image.colourFilter(2, 0)

    image.showImage()

    image.saveImage(save_path)


main()