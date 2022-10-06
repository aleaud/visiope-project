import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class DeepGlobeDataset(Dataset):
  def __init__(self, df, color_dict, test=False, transforms=None):
    super().__init__()
    self.df = df
    self.color_dict = color_dict
    self.test = test
    self.transforms = transforms
    
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    data = self.df.iloc[idx]

    image = Image.open(data.images)
    mask = Image.open(data.masks)

    if self.transforms is not None:
      image = self.transforms(image)
      mask = self.transforms(mask)

    image = np.asarray(image)
    mask = np.asarray(mask)

    if not self.test:
      t = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
      image = t(image)
    
    mask = self.rgb2category(mask)
    mask = np.expand_dims(mask, axis=0)
    #mask = torch.from_numpy(mask).long()
    mask = torch.Tensor(mask).long()

    return image, mask

  #return the color-text version of the mask, given the RGB mask
  def rgb2category(self, rgb_mask):
      category_mask = np.zeros(rgb_mask.shape[:2], dtype=np.int8)
      for i, row in self.color_dict.iterrows():
          category_mask += (np.all(rgb_mask.reshape((-1, 3)) == (row['r'], row['g'], row['b']), axis=1).reshape(rgb_mask.shape[:2]) * i)
      return category_mask

  #return the RGB version of the mask, given the color-text version
  def category2rgb(self, category_mask):
      rgb_mask = np.zeros(category_mask.shape[:2] + (3,))
      for i, row in self.color_dict.iterrows():
          rgb_mask[category_mask==i] = (row['r'], row['g'], row['b'])
      return np.uint8(rgb_mask)
  


