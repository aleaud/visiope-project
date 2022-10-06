import numpy as np
import pandas as pd
import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.utils import shuffle


# ---- CONSTANTS ---- #
DEEPGLOBE_ROOT_DIR = "/home/aleaud/Documents/visiope/datasets/deepglobe"
CLASS_DICT_PATH = os.path.join(DEEPGLOBE_ROOT_DIR, 'class_dict.csv')
TRAIN_DIR = os.path.join(DEEPGLOBE_ROOT_DIR, 'train')
#VALID_DIR = os.path.join(ROOT_DIR, 'valid')
#TEST_DIR = os.path.join(ROOT_DIR, 'test')
IMG_FILE_EXT = '.jpg'
MASK_FILE_EXT = '.png'
IMAGE_SIZE = 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#SAVE_CUSTOM_MODEL_PATH = "/home/aleaud/Documents/visiope/saved_models/unet-deepglobe.pt"
#SAVE_PRETRAIN_MODEL_PATH = "/home/aleaud/Documents/visiope/saved_models/unet-pretrain-deepglobe.pt"


# ---- FUNCTIONS ---- #
#returns a DataFrame containing the class name and the color associated to it, divided in RGB values
def decode_class_colors(filepath):
    color_dict = pd.read_csv(filepath)
    return color_dict

#returns a DataFrame referring to the training set containing absolute paths to
#the training images and the associated semantic maps (only for DeepGlobe)
def extract_data_df(root):
    data = {'images': [], 'masks': []}
    
    for _, _, files in os.walk(root):
        for file in files:
            filename = os.path.join(TRAIN_DIR, file)
            if IMG_FILE_EXT in filename:
                data['images'].append(filename)
            elif MASK_FILE_EXT in filename:
                data['masks'].append(filename)
        
        #align data
        data['images'].sort(); data['masks'].sort()
    
    return shuffle(pd.DataFrame(data=data, index=np.arange(0, len(data['images']))))

#show some samples from a batch. If 'range' is True, show the whole batch -> DON'T WORK :(
def show_samples(dl, idx=0, range=False):
    #choose a random sample in the batch if idx exceeds the limits (0 to batch_size-1)
    for _, (images, masks) in enumerate(dl):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2.set_xticks([]); ax2.set_yticks([])

        if range:
            ax1.set_title('IMAGES')
            ax1.imshow(make_grid(images.permute(1, 2, 0), nrow=dl.batch_size))
            ax2.set_title('MASKS')
            ax2.imshow(make_grid(masks.permute(1, 2, 0), nrow=dl.batch_size))
        else:
            #choose a random sample in the batch if idx exceeds the limits (0 to batch_size-1)    
            idx = random.randint(0, dl.batch_size) if idx > len(dl) else idx
            img = images[idx].detach().cpu()
            mask = masks[idx].detach().cpu()
            ax1.set_title('IMAGE' + str(idx))
            ax1.imshow(img.permute(1, 2, 0))
            ax2.set_title('MASK' + str(idx))
            ax2.imshow(mask.permute(1, 2, 0).squeeze())
        
        break


#functions for plotting results
#plot loss
def plot_loss(history, save=False, path_to_save='/home/aleaud/Pictures/visiope_pics/loss.png'):
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['valid_loss'], label='valid')
    plt.title('Loss per epoch')
    plt.xlabel('epoch'); plt.ylabel('loss')
    plt.legend(), plt.show()
    if save:
        plt.savefig(path_to_save, bbox_inches='tight')
        print("Plot of losses saved in {}".format(path_to_save))

#plot accuracy
def plot_accuracy(history, save=False, path_to_save='/home/aleaud/Pictures/visiope_pics/acc.png'):
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['valid_acc'], label='valid_acc')
    plt.title('Accuracy per epoch')
    plt.xlabel('epoch'); plt.ylabel('accuracy')
    plt.legend(), plt.show()
    if save:
        plt.savefig(path_to_save, bbox_inches='tight')
        print("Plot of accuracy saved in {}".format(path_to_save))

#plot IoU -> not used
def plot_iou(history, save=False, path_to_save='/home/aleaud/Pictures/visiope_pics/iou.png'):
    plt.plot(history['train_iou'], label='train_iou')
    plt.plot(history['valid_iou'], label='valid_iou')
    plt.title('IoU per epoch')
    plt.xlabel('epoch'); plt.ylabel('IoU')
    plt.legend(), plt.show()
    if save:
        plt.savefig(path_to_save, bbox_inches='tight')
        print("Plot of IoU saved in {}".format(path_to_save))

