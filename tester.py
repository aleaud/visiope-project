import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from utils import DEVICE
from metrics import pixel_accuracy, mIoU


def predict_image_mask_miou(model, image, mask):
    t = T.Compose([ T.ToTensor(), 
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = t(image)
    image=image.to(DEVICE); mask = mask.to(DEVICE)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def predict_image_mask_pixel(model, image, mask):
    t = T.Compose([ T.ToTensor(), 
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = t(image)
    image=image.to(DEVICE); mask = mask.to(DEVICE)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc

def miou_score(model, test_set):
    score_iou = []
    for i in range(len(test_set)):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou

def pixel_acc(model, test_set):
    accuracy = []
    for i in range(len(test_set)):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        accuracy.append(score)
    return accuracy

def test(model, test_set, show=False):
    torch.cuda.empty_cache()
    model.to(DEVICE)
    #start = time.time()

    model.eval()
    iou = miou_score(model, test_set)
    acc = pixel_acc(model, test_set)
    print('Test mIoU: {}, Test pixel accuracy: {}'.format(np.mean(iou), np.mean(acc)))
    if show:
        print_test_sample(model, test_set, how_many=3)
    return acc, iou        

def print_test_sample(model, test_set, how_many=2):
    #take two random samples from the test set, otherwise take the first two samples
    r = random.sample(range(len(test_set)), how_many) if how_many > 2 else [0,1]

    for idx in r:
        
        image, mask = test_set[idx]
        iou_mask, iou_score = predict_image_mask_miou(model, image, mask)
        pixel_mask, pixel_score = predict_image_mask_pixel(model, image, mask)
        
        image = Image.fromarray(image)
        mask = Image.fromarray(test_set.category2rgb(torch.squeeze(mask)))
        iou_mask = Image.fromarray(test_set.category2rgb(iou_mask))
        pixel_mask = Image.fromarray(test_set.category2rgb(pixel_mask))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,10))
        ax1.imshow(image)
        ax1.set_title('Picture' + str(idx))

        ax2.imshow(mask)
        ax2.set_title('Ground truth')
        ax2.set_axis_off()

        ax3.imshow(iou_mask)
        ax3.set_title('Prediction | mIoU {:.3f}'.format(iou_score))
        ax3.set_axis_off()

        ax4.imshow(pixel_mask)
        ax4.set_title('Prediction | pixel accuracy {:.3f}'.format(pixel_score))
        ax4.set_axis_off()

        plt.show()
