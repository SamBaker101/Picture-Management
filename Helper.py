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
        try:
            self.image = Image.open(path)
            self.width, self.height = self.image.size
            self.tensor = ToTensor(self.image)
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
        new_tensor = torch.zeros(3, int((self.width)/2), int((self.height)/2))

        self.tensor = ToTensor(self.image)
        for i in range(3):
            for j in range(int(self.width/2 - 1)):
                for k in range(int(self.height/2 - 1)):
                    new_tensor[i][j][k] = self.tensor[i][j*2][k*2]
        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size

    def halfResHigh(self):
        new_tensor = torch.zeros(3, int(self.width / 2), int(self.height / 2))
        view_tensor = torch.zeros(2, 2)

        self.tensor = ToTensor(self.image)
        for i in range(3):
            for j in range(int(self.width / 2 - 1)):
                for k in range(int(self.height / 2 - 1)):
                    view_tensor[0] = self.tensor[i][j*2][k*2:k*2+2]
                    view_tensor[1] = self.tensor[i][j*2+1][k*2:k*2+2]
                    new_tensor[i][j][k] = torch.max(view_tensor)
        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size

    def halfResLow(self):
        new_tensor = torch.zeros(3, int(self.width / 2), int(self.height / 2))
        view_tensor = torch.zeros(2, 2)

        self.tensor = ToTensor(self.image)
        for i in range(3):
            for j in range(int(self.width / 2 - 1)):
                for k in range(int(self.height / 2 - 1)):
                    view_tensor[0] = self.tensor[i][j*2][k*2:k*2+2]
                    view_tensor[1] = self.tensor[i][j*2+1][k*2:k*2+2]
                    new_tensor[i][j][k] = torch.min(view_tensor)
        self.tensor = new_tensor
        self.image = ToPIL(self.tensor)
        self.width, self.height = self.image.size