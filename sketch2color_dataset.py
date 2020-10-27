from torch.utils.data import Dataset
import os
from os import listdir
from torchvision import transforms
import cv2
from albumentations import (
    HorizontalFlip, RandomResizedCrop, Resize, OneOf, Compose, RandomBrightnessContrast, HueSaturationValue
)
from albumentations.pytorch.transforms import ToTensor


def get_train_transform(input_height, input_width):
    return Compose([
        RandomBrightnessContrast(),
        HueSaturationValue()
    ]),\
    Compose([
        OneOf([
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_NEAREST),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_LINEAR),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_CUBIC),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_AREA),
            RandomResizedCrop(input_height, input_width, interpolation=cv2.INTER_LANCZOS4),
            Resize(input_height, input_width)
        ], p=1.0),
        HorizontalFlip(),
        ToTensor()
    ])


def get_test_transform(input_height, input_width):
    return Compose([
        Resize(input_height, input_width),
        ToTensor()
    ])

class Sketch2ColorDataset(Dataset):
    def __init__(self, dataset, phase, input_height, input_width, processed_dir):
        super().__init__()
        self.phase = phase

        assert phase in ['train', 'val', 'test']
        
        sketch_dir = os.path.join(processed_dir, dataset, phase, 'sketch')
        color_dir = os.path.join(processed_dir, dataset, phase, 'color')
        
        sketch_fnames = sorted(listdir(sketch_dir), key=lambda x:int(x.split('.')[0]))
        color_fnames = sorted(listdir(color_dir), key=lambda x:int(x.split('.')[0]))

        self.sketch_fnames = [os.path.join(sketch_dir, fname) for fname in sketch_fnames]
        self.color_fnames = [os.path.join(color_dir, fname) for fname in color_fnames]

        if phase == 'train':
            transform = get_train_transform(input_height, input_width)
        else:
            transform = get_test_transform(input_height, input_width)
        self.transform = Compose(transform, additional_targets={'image2': 'image'})

    def __getitem__(self, idx):
        """
        output:
        sketch image: B x 1 x H x W image tensor scaled to [0, 1]
        color image: B x 3 x H x W image tensor scaled to [0, 1]
        """
        sketch_img = cv2.cvtColor(cv2.imread(self.sketch_fnames[idx]), cv2.COLOR_BGR2RGB)
        color_img = cv2.cvtColor(cv2.imread(self.color_fnames[idx]), cv2.COLOR_BGR2RGB)

        if self.phase == 'train':
            transform1, transform2 = self.transform
            color_img = transform1(image=color_img)['image']
            aug_output = transform2(image=sketch_img, image2=color_img)
            out1, out2 = aug_output['image'], aug_output['image2']
        else:
            aug_output = self.transform(image=sketch_img, image2=color_img)
            out1, out2 = aug_output['image'], aug_output['image2']
        
        # RGB to GRAY
        out1 = 0.299 * out1[0:1,:,:] + 0.587 * out1[1:2,:,:] + 0.114 * out1[2:3,:,:]

        return out1, out2

    def __len__(self):
        return len(self.sketch_fnames)
