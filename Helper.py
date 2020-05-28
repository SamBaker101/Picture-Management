'''
Sam Baker - 05/15/2020
deer picture source: personal
jazz picture source: https://www.flickr.com/photos/29713047@N00/157845230
'''

from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt

ToTensor = torchvision.transforms.ToTensor()
ToPIL = torchvision.transforms.ToPILImage()

class Picture:
    def __init__(self, path):
        self.image = 0
        self.width, self.height = 0, 0
        self.tensor = 0
        self.flat_tensor = 0
        try:
            self.image = Image.open(path)
            self.width, self.height = self.image.size
            self.tensor = ToTensor(self.image)
            self.flat_tensor = torch.flatten(self.tensor)
        except IOError:
            print('Something went wrong')

    def getImage(self):
        return self.image

    def saveImage(self, path):
        self.image.save(path)

    def showImage(self):
        self.image.show()

    def cropImage(self, x, y, size_x, size_y):
        # x,y refer to top left corner
        self.image = self.image.crop((x, y, x + size_x, y + size_y))
        self.tensor = ToTensor(self.image)
        self.width, self.height = self.image.size

    def imageToTensor(self):
        self.tensor = ToTensor(self.image)

    def tensorToImage(self):
        self.image = ToPIL(self.tensor)

    def flatten(self):
        self.flat_tensor = torch.flatten(self.tensor)

    def plotTensor(self, channel):
        plt.figure()
        plt.imshow(self.tensor[channel].numpy())
        plt.colorbar()
        plt.show()

    def colourFilter(self, colour, new_value):
        # Replaces all values in chosen colour layer with provided value
        for i in range(0, self.tensor[colour].size()[0]):
            for j in range(0, self.tensor[colour].size()[1]):
                self.tensor[colour][i][j] = new_value
        self.tensorToImage()

    def halfResLazy(self):
        new_tensor = torch.zeros(3, int((self.height)/2), int((self.width)/2))

        self.tensor = ToTensor(self.image)
        for i in range(3):
            for j in range(int(self.width/2 - 1)):
                for k in range(int(self.height/2 - 1)):
                    new_tensor[i][k][j] = self.tensor[i][k*2][j*2]
        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size

    def halfResHigh(self):
        new_tensor = torch.zeros(3, int((self.height)/2), int((self.width)/2))
        view_tensor = torch.zeros(2, 2)

        self.tensor = ToTensor(self.image)
        for i in range(3):
            for j in range(int(self.width / 2 - 1)):
                for k in range(int(self.height / 2 - 1)):
                    view_tensor[0] = self.tensor[i][k * 2][j * 2:j * 2 + 2]
                    view_tensor[1] = self.tensor[i][k * 2 + 1][j * 2:j * 2 + 2]
                    new_tensor[i][k][j] = torch.max(view_tensor)
        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size

    def halfResLow(self):
        new_tensor = torch.zeros(3, int((self.height)/2), int((self.width)/2))
        view_tensor = torch.zeros(2, 2)

        self.tensor = ToTensor(self.image)
        for i in range(3):
            for j in range(int(self.width / 2 - 1)):
                for k in range(int(self.height / 2 - 1)):
                    view_tensor[0] = self.tensor[i][k*2][j*2:j*2+2]
                    view_tensor[1] = self.tensor[i][k*2+1][j*2:j*2+2]
                    new_tensor[i][k][j] = torch.min(view_tensor)
        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size

    def grayScale(self):
        self.tensor = ToTensor(self.image)
        new_tensor = torch.zeros(1, self.height, self.width)

        for i in range(self.height):
            for j in range(self.width):
                new_tensor[0][i][j] = (self.tensor[0][i][j] +
                                       self.tensor[1][i][j] +
                                       self.tensor[2][i][j])/3
        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)

    def maxPool(self, kernel_size=2, stride=1):
        if stride > kernel_size:
            print('Invalid Parameters for Pooling')
            return

        if stride==1:
            new_tensor = torch.zeros(3, int((self.height)-1), int((self.width)-1))

        else:
            new_tensor = torch.zeros(3, int(self.height/stride), int(self.width/stride))

        view_tensor = torch.zeros(2, 2)
        self.tensor = ToTensor(self.image)

        for i in range(3):
            j , k = 0, 0
            while j < self.width - kernel_size:
                while k < self.height - kernel_size:
                    view_tensor[0] = self.tensor[i][k][j:j + 2]
                    view_tensor[1] = self.tensor[i][k + 1][j:j+ 2]
                    new_tensor[i][int(k/stride)][int(j/stride)] = torch.max(view_tensor)
                    k += stride
                j += stride
                k = 0

        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size

    def mmFilter(self, filter):
        filter_w, filter_h = filter.size()

        view_tensor = torch.zeros(filter_w, filter_h)
        j, k = int((self.width % filter_w) / 2), int((self.height % filter_h) / 2)

        for i in range(3):
            while j < self.width - 1 - int((self.width % filter_w) / 2):
                while k < self.height - 1 - int((self.height % filter_h) / 2):
                    for m in range(filter_h):
                        view_tensor[m] = self.tensor[i][k + m][j:j + int(filter_w)]

                    view_tensor = torch.mm(filter, view_tensor)

                    for m in range(filter_h):
                        self.tensor[i][k + m][j:j + int(filter_w)] = view_tensor[m]

                    k += filter_h

                j += filter_w
                k = int((self.height - self.height % filter_h) / 2 - 1)

    def conv1(self, filter, stride=1):
        filter_w, filter_h = filter.size()
        if stride > filter_h or stride > filter_w:
            print('Invalid Parameters for Pooling')
            return

        if stride==1:
            new_tensor = torch.zeros(3, int((self.height) - 1 - filter_h), int((self.width) - 1 - filter_w))

        else:
            new_tensor = torch.zeros(3, int(self.height/stride - filter_h), int(self.width/stride - filter_w))

        _, new_tensor_h, new_tensor_w = new_tensor.size()

        view_tensor = torch.zeros(filter_w, filter_h)
        self.tensor = ToTensor(self.image)

        for i in range(3):
            j , k = 0, 0
            while k < new_tensor_h - 1:
                while j < new_tensor_w - 1:
                    for m in range(filter_h):
                        view_tensor[m] = self.tensor[i][k * stride + m][j * stride:j*stride + int(filter_w)]

                    new_tensor[i][k][j] = torch.dot(torch.flatten(view_tensor), torch.flatten(filter))
                    j += 1
                k += 1
                j = 0

        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size

    def relu1(self):
        for i in range(3):
            for j in range(int(self.width  - 1)):
                for k in range(int(self.height  - 1)):
                    self.tensor[i][k][j] = max(0, self.tensor[i][k][j])

    def sigmoid(self):
        for i in range(3):
            for j in range(int(self.width - 1)):
                for k in range(int(self.height  - 1)):
                    self.tensor[i][k][j] = 1/(1+torch.exp(-self.tensor[i][k][j]))


class Filters:
    def __init__(self):
        self.filter = 0

    def vertLine22(self):
        self.filter = torch.tensor([[-1., 1.], [-1., 1.]])
        return self.filter

    def vertLine33(self):
        self.filter = torch.tensor([[0, -1., 1.], [0, -1., 1.], [0, -1., 1.]])
        return self.filter

    def horLine22(self):
        self.filter = torch.tensor([[1., 1.], [-1., -1.]])
        return self.filter

    def horLine33(self):
        self.filter = torch.tensor([[0, 0, 0], [-1., -1., -1.], [1., 1., 1.]])
        return self.filter

    def edge33(self):
        self.filter = torch.tensor([[0, 1., 0], [1., -4., 1.], [0, 1., 0]])
        return self.filter

    def edge55(self):
        self.filter = torch.tensor([[0, 0, 1., 0, 0],
                                    [0, 1., -2., 1., 0],
                                    [1., -2., 0, -2., 1.],
                                    [0, 1., -2., 1., 0],
                                    [0, 0, 1., 0, 0]])

        return self.filter

    def x99(self):
        self.filter = torch.tensor([[1., 0, 0, 0, -1., 0, 0, 0, 1.],
                                    [0, 1., 0, 0, -1., 0, 0, 1., 0],
                                    [0, 0, 1., 0, -1., 0, 1., 0, 0],
                                    [0, 0, 0, 1., -1., 1., 0, 0, 0],
                                    [-1., -1., -1., -1., 0, -1., -1., -1., -1.],
                                    [0, 0, 0, 1., -1., 1., 0, 0, 0],
                                    [0, 0, 1., 0, -1., 0, 1., 0, 0],
                                    [0, 1., 0, 0, -1., 0, 0, 1., 0],
                                    [1., 0, 0, 0, -1., 0, 0, 0, 1.]])
        return self.filter