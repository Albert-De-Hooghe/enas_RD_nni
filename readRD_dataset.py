# fonction pour lire le dataset RD ??
from PIL import Image
import os
import os.path
import pandas as pd
from os import path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# from .vision import VisionDataset
# from .utilsFromTorchvisionDatasets import check_integrity, download_and_extract_archive
path_of_image_folder = "/home/albert/darts_data_nni/preprocessedTrain224"
print(path_of_image_folder)
print(os.path.dirname( __file__ ))
#"/home/albert/Bureau/preprocessedTrain224"
#
def returnListOfClassesOnValidation():## dans le search on veut savoir combien de classes 0 on été envoyé
    # car le WeightedKappa change
    var1, var2 = 30001, 35120 # search

    compteur_0 = 0
    compteur_1 = 0
    compteur_2 = 0
    compteur_3 = 0
    compteur_4 = 0

    train_csv = pd.read_csv('trainLabels.csv')
    for i in range(var1, var2):
        if path.exists(path_of_image_folder+'/'+train_csv.image[i]+'.png'):
            if train_csv.level[i] == 0:
                compteur_0 += 1
            elif train_csv.level[i] == 1:
                compteur_1 += 1
            elif train_csv.level[i] == 2:
                compteur_2 += 1
            elif train_csv.level[i] == 3:
                compteur_3 += 1
            elif train_csv.level[i] == 4:
                compteur_4 += 1
            else:
                print("what ??")
    return [compteur_0, compteur_1, compteur_2, compteur_3, compteur_4]

def returnListOfClassesOnTrain():## dans le search on veut savoir combien de classes 0 on été envoyé
    # car le WeightedKappa change
    var1 = 30000 # search

    compteur_0 = 0
    compteur_1 = 0
    compteur_2 = 0
    compteur_3 = 0
    compteur_4 = 0

    train_csv = pd.read_csv('trainLabels.csv')
    for i in range(var1):
        if path.exists(path_of_image_folder+'/'+train_csv.image[i]+'.png'):
            if train_csv.level[i] == 0:
                compteur_0 += 1
            elif train_csv.level[i] == 1:
                compteur_1 += 1
            elif train_csv.level[i] == 2:
                compteur_2 += 1
            elif train_csv.level[i] == 3:
                compteur_3 += 1
            elif train_csv.level[i] == 4:
                compteur_4 += 1
            else:
                print("what ??")
    return [compteur_0, compteur_1, compteur_2, compteur_3, compteur_4]

def RD_Dataset_train_binary_classif(taille):

    train_csv = pd.read_csv('trainLabels.csv')
    list = []
    #print(train_csv)
    if taille == "big":
        var1 = 15000
    elif taille == "small":
        var1 = 1500
    elif taille == "mini":
        var1 = 15
    for i in range(var1):    # len(train_csv.image)
        if path.exists(path_of_image_folder+'/'+train_csv.image[i]+'.png'):
            list.append([getTensorfromImagePath(path_of_image_folder + '/' + train_csv.image[i]+'.png'), zeroOrOneForTheLabel(train_csv.level[i])]) #])#, train_csv.image[i]])


    # function that return the list of (image, labels) from the path of the folder (image in the tensor format)

    # list_image_name = os.listdir(path_of_image_folder)
    # for single_image_name in list_image_name[0:7]:
    #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))

    return list
# def RD_Dataset_train():
#     path_of_image_folder = "./preprocessedTrain"
#     train_csv = pd.read_csv('trainLabels.csv')
#     list = np.array([])
#     #print(train_csv)
#     for i in range(1000):    # len(train_csv.image)
#         if path.exists(path_of_image_folder+'/'+train_csv.image[i]+'.png'):
#             np.append(list,[getTensorfromImagePath(path_of_image_folder + '/' + train_csv.image[i]+'.png'), zeroOrOneForTheLabel(train_csv.level[i])])
#
#
#     # function that return the list of (image, labels) from the path of the folder (image in the tensor format)
#
#     # list_image_name = os.listdir(path_of_image_folder)
#     # for single_image_name in list_image_name[0:7]:
#     #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))
#
#     return list


def RD_Dataset_test_binary_classif(taille):
    if taille == "big":
        var1, var2 = 15001, 21400
    elif taille == "small":
        var1, var2 = 1501, 2100
    elif taille == "mini":
        var1, var2 = 16, 30


    train_csv = pd.read_csv('trainLabels.csv')
    list = []
    #print(train_csv)
    for i in range(var1,var2):    # len(train_csv.image)
        if path.exists(path_of_image_folder+'/'+train_csv.image[i]+'.png'):
            list.append([getTensorfromImagePath(path_of_image_folder + '/' + train_csv.image[i]+'.png'), zeroOrOneForTheLabel(train_csv.level[i])])


    # function that return the list of (image, labels) from the path of the folder (image in the tensor format)

    # list_image_name = os.listdir(path_of_image_folder)
    # for single_image_name in list_image_name[0:7]:
    #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))

    return list


def RD_Dataset_valid_binary_classif(taille):
    if taille == "big":
        var1, var2 = 21401, 27000
    elif taille == "small":
        var1, var2 = 2101, 2500
    elif taille == "mini":
        var1, var2 = 31, 45
     

    train_csv = pd.read_csv('trainLabels.csv')
    list = []
    #print(train_csv)
    for i in range(var1, var2):    # len(train_csv.image)
        if path.exists(path_of_image_folder+'/'+train_csv.image[i]+'.png'):
            list.append([getTensorfromImagePath(path_of_image_folder + '/' + train_csv.image[i]+'.png'), zeroOrOneForTheLabel(train_csv.level[i])])


    # function that return the list of (image, labels) from the path of the folder (image in the tensor format)

    # list_image_name = os.listdir(path_of_image_folder)
    # for single_image_name in list_image_name[0:7]:
    #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))

    return list


def RD_Dataset_train_5_classes(taille):

    train_csv = pd.read_csv('trainLabels.csv')
    list = []
    # print(train_csv)
    if taille == "big":
        var1 = 15000
    elif taille == "search":
        var1 = 30000
    elif taille == "small":
        var1 = 500
    elif taille == "mini":
        var1 = 50
    for i in range(var1):  # len(train_csv.image)
        if path.exists(path_of_image_folder + '/' + str(train_csv.image[i]) + '.png'):
            list.append([getTensorfromImagePath(path_of_image_folder + '/' + str(train_csv.image[i]) + '.png'),
                         train_csv.level[i]])

    # function that return the list of (image, labels) from the path of the folder (image in the tensor format)

    # list_image_name = os.listdir(path_of_image_folder)
    # for single_image_name in list_image_name[0:7]:
    #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))

    return list


# def RD_Dataset_train():
#     path_of_image_folder = "./preprocessedTrain"
#     train_csv = pd.read_csv('trainLabels.csv')
#     list = np.array([])
#     #print(train_csv)
#     for i in range(1000):    # len(train_csv.image)
#         if path.exists(path_of_image_folder+'/'+train_csv.image[i]+'.png'):
#             np.append(list,[getTensorfromImagePath(path_of_image_folder + '/' + train_csv.image[i]+'.png'), zeroOrOneForTheLabel(train_csv.level[i])])
#
#
#     # function that return the list of (image, labels) from the path of the folder (image in the tensor format)
#
#     # list_image_name = os.listdir(path_of_image_folder)
#     # for single_image_name in list_image_name[0:7]:
#     #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))
#
#     return list


def RD_Dataset_valid_5_classes(taille):
    if taille == "big":
        var1, var2 = 15001, 21400
    elif taille == "search":
        var1, var2 = 30001, 35119
    elif taille == "small":
        var1, var2 = 501, 1000
    elif taille == "mini":
        var1, var2 = 51, 100

    train_csv = pd.read_csv('trainLabels.csv')
    list = []
    # print(train_csv)
    for i in range(var1, var2):  # len(train_csv.image)
        if path.exists(path_of_image_folder + '/' + str(train_csv.image[i]) + '.png'):
            list.append([getTensorfromImagePath(path_of_image_folder + '/' + str(train_csv.image[i]) + '.png'),
                         train_csv.level[i]])

    # function that return the list of (image, labels) from the path of the folder (image in the tensor format)

    # list_image_name = os.listdir(path_of_image_folder)
    # for single_image_name in list_image_name[0:7]:
    #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))

    return list

class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.image_files)

    def __getitem__(self, index):
        if self.train:
            image_file, label = self.data.iloc[index]
        else:
            # if test simply return -1 for label, I do this in order to
            # re-use same dataset class for test set submission later on
            image_file, label = self.image_files[index], -1
            image_file = image_file.replace(".jpeg", "")

        image = np.array(Image.open(os.path.join(self.images_folder, image_file)))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, image_file

def RD_Dataset_test_5_classes(taille):
    if taille == "big":
        var1, var2 = 21401, 27000
    elif taille == "small":
        var1, var2 = 1001, 2000
    elif taille == "mini":
        var1, var2 = 101, 155

    train_csv = pd.read_csv('trainLabels.csv')
    list = []
    # print(train_csv)
    for i in range(var1, var2):  # len(train_csv.image)
        if path.exists(path_of_image_folder + '/' + str(train_csv.image[i]) + '.png'):
            list.append([getTensorfromImagePath(path_of_image_folder + '/' + str(train_csv.image[i]) + '.png'),
                         train_csv.level[i]])

    # function that return the list of (image, labels) from the path of the folder (image in the tensor format)

    # list_image_name = os.listdir(path_of_image_folder)
    # for single_image_name in list_image_name[0:7]:
    #     list.append(getTensorfromImagePath(path_of_image_folder+'/'+single_image_name))

    return list

def zeroOrOneForTheLabel(entry):
    if entry >= 3:
        return 1
    else:
        return 0

def getLabelfromImageName(): ## fonctionne que pour les données de train
    train_csv = pd.read_csv('trainLabels.csv')

def getTensorfromImagePath(image_path):
    # transform1 = transforms.PILToTensor()
    #
    # with Image.open(image_path) as im:
    #     tensor = transform1(np.array(im)) # / 255  # normalisation
    # return tensor

    ## modification du code du dessus pour résoudre l'erreur https://stackoverflow.com/questions/43268156/process-finished-with-exit-code-137-in-pycharm
    with Image.open(image_path) as im:
        tensor = transforms.ToTensor()(np.array(im))# / 255  # normalisation
    return tensor


if __name__ == "__main__":
    # img = Image.open("./testPreprocessed/11_left.png")

    print(returnListOfClassesOnValidation()) ## donne [3827, 364, 724, 103, 101]
    #print(returnListOfClassesOnTrain())
    # train_csv = pd.read_csv('trainLabels.csv')

    # print(len(train_csv.level))

    # getTensorfromImagePath("./testPreprocessed/11_left.png").shape# .shape nous donne torch.Size([3, 448, 448])

    # print(transform2(transform1(img)))

    # with Image.open("./testPreprocessed/11_left.png") as im:
    #     # im.show()
    #     print(transform1(im)[1][100]/255)  ## a diviser par 255
