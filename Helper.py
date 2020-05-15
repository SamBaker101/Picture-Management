from PIL import Image
import torch
import torchvision

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
        except IOError:
            print('Something went wrong')


    def getImage(self):
        return self.image

    def saveImage(self, path):
        self.image.save(path)

    def showImage(self):
        self.image.show()

    def cropImage(self, x, y, size_x, size_y):
        #x,y refer to top left corner
        self.image = self.image.crop((x, y, x + size_x, y + size_y))

    def imageToTensor(self):
        self.tensor = ToTensor(self.image)

    def tensorToImage(self):
        self.image = ToPIL(self.tensor)

    def colourFilter(self, colour, new_value):
        for i in range(0, self.tensor[colour].size()[0]):
            for j in range(0, self.tensor[colour].size()[1]):
                self.tensor[colour][i][j] = new_value
        self.tensorToImage()

