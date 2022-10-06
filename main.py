import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from PIL import Image
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix

from utils import *
from trainer import *
from metrics import *
from tester import test
from dataset import DeepGlobeDataset
from model import CustomUNet


#set state for reproducibility
seed = 24
torch.manual_seed(seed)
np.random.seed(seed)


#       ---------   DEEPGLOBE   ----------      #

start = time.time()

#transforms
t_train = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST),
                        T.RandomHorizontalFlip(), 
                        T.RandomVerticalFlip()])
t_val = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST),
                        T.RandomHorizontalFlip()])
t_test = T.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)


#extract data
print('Decoding color dictionary...')
color_dict = decode_class_colors(CLASS_DICT_PATH)
#print(color_dict)
print('Extracting data...')
data = extract_data_df(TRAIN_DIR)
#print(data.head())
#sample = data.loc[0]
#print("Example of image ID: {}\nExample of mask ID: {}".format(sample['images'], sample['masks']))

#split data 
X_trainval, X_test = train_test_split(data, test_size=0.2, random_state=seed)
X_train, X_val = train_test_split(X_trainval, test_size=0.4, random_state=seed)
#print("Shapes:\n\t-> X_train: {}\n\t-> X_val: {}\n\t-> X_test: {}".format(X_train.shape, X_val.shape, X_test.shape))

#datasets 
print('Loading datasets...')
train_set = DeepGlobeDataset(X_train, color_dict, test=False, transforms=t_train)
val_set = DeepGlobeDataset(X_val, color_dict, test=False, transforms=t_val)
test_set = DeepGlobeDataset(X_test, color_dict, test=True, transforms=t_test)
#print("Shapes:\n\t-> train_set: {}\n\t-> val_set: {}\n\t-> test_set: {}".format(train_set, val_set, test_set))

#dataloaders 
batch_size = 8
print('Configuring dataloaders...')
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=True)

#show a sample in a batch...
#print('Show a sample in the training and validation batch...')
#show_samples(train_dl)
#show_samples(val_dl)
# ... or an entire batch
#print('Show an entire batch...')
#show_samples(train_dl, range=True)
#show_samples(val_dl, range=True)


#UNet takes 3 channels (RGB images) and outputs prediction maps for the 7 output classes
in_channels = 3
out_channels = len(color_dict)
model = CustomUNet(in_channels, out_channels)


#train -> try different settings!
#epochs = [10, 20, 30, 40, 50]
#lrs = [1e-3, 0.001, 0.01]
epochs = 20
lr = 1e-2
weight_decay = 1e-3

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
print("Starting training...")
#history = train(model, train_dl, val_dl, loss_fn, optimizer, pixel_accuracy, DEVICE, epochs=1)                              # 'DEBUG' TRAIN
history = train(model, train_dl, val_dl, loss_fn, optimizer, pixel_accuracy, DEVICE, epochs=epochs)
#print(history)

#plot results on custom model
print('Show results of custom UNet...')
plot_loss(history)
plot_accuracy(history)
plot_iou(history)


#save model
#print('Saving the model parameters...')
#torch.save(model.state_dict, SAVE_CUSTOM_MODEL_PATH)


#test
print('Testing the model...')
test(model, test_set, show=True)

#operation on pre-trained model
print('Training on standard UNet pre-trained on ImageNet...')
pretrained_model = smp.Unet('resnet34', encoder_weights='imagenet', classes=out_channels, activation='softmax', encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=lr, weight_decay=weight_decay)
#history_pretrained = train(pretrained_model, train_dl, val_dl, loss_fn, optimizer, pixel_accuracy, DEVICE, epochs=1)        # 'DEBUG' TRAIN
history_pretrained = train(pretrained_model, train_dl, val_dl, loss_fn, optimizer, pixel_accuracy, DEVICE, epochs=epochs)
#print(history_pretrained)
print('Testing on pre-trained UNet...')
test(pretrained_model, test_set, show=True)


#plot results on pre-trained model
print('Show results on pre-trained UNet...')
plot_loss(history_pretrained)
plot_accuracy(history_pretrained)
plot_iou(history_pretrained)

#save model
#print('Saving the model parameters...')
#torch.save(model.state_dict, SAVE_PRETRAIN_MODEL_PATH)

how_long = time.time() - start
print("End of the project, run in {:.0f}m {:.0f}s. Hope you've enjoyed it :)".format(how_long // 60, how_long % 60)) 