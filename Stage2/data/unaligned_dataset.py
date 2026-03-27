import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
import cv2

def get_transform2():
    transform_list = []
    transform_list.append(transforms.Resize((256), transforms.InterpolationMode.BICUBIC))
    transform_list.append(transforms.ToTensor())
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_params(size):
    new_h = new_w = 286
    x = random.randint(0, np.maximum(0, new_w - 256))
    y = random.randint(0, np.maximum(0, new_h - 256))
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}
def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img
def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
def transform(params=None,  method=transforms.InterpolationMode.BICUBIC):
    transform_list = []
    osize = [286,286]
    transform_list.append(transforms.Resize(osize, method))
    if params is None:
        transform_list.append(transforms.RandomCrop(256))
    else:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 256)))
    if params is None:
        transform_list.append(transforms.RandomHorizontalFlip())
    elif params['flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    transform_list += [transforms.ToTensor()]      
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        print(opt.max_dataset_size,opt.dataroot ,self.dir_A, self.dir_B)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        self.opt.lab = opt.lab
        self.opt.phase = opt.phase
        self.opt.epoch = opt.epoch
        self.lower_purple = np.array([0, 0, 0], dtype=np.uint8)
        self.upper_purple = np.array([150, 100, 180], dtype=np.uint8)  
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        D_img = np.array(B_img)
        D_img = 255-  D_img 
        D_img = Image.fromarray(D_img)
        
        # C_img = A_img
        C_img = np.array(A_img)
        purple_mask = cv2.inRange(C_img, self.lower_purple, self.upper_purple)  
        
        
        points = np.column_stack(np.where(purple_mask > 0))

        # 遍历每个点，将附近十个像素也变为 222
        for point in points:
            x, y = point
            for i in range(max(0, x - 3), min(C_img.shape[0], x + 4)):
                for j in range(max(0, y - 3), min(C_img.shape[1], y + 4)):
                    C_img[i, j] = [222, 222, 222]
        # C_img[purple_mask > 0] = [222, 222, 222]
        C_img = Image.fromarray(C_img)
        
        # apply image transformation
        if self.opt.phase =='train':
            transform_params = get_params( 256)
            A_transform = transform( transform_params)
            B_transform = transform( transform_params)
            C_transform = transform( transform_params)
            D_transform = transform( transform_params)
        else:
            A_transform = get_transform2( )
            B_transform = get_transform2( )
            C_transform = get_transform2( )
            D_transform = get_transform2( )
        A = A_transform(A_img)
        B = B_transform(B_img)
        C = C_transform(C_img)
        D = D_transform(D_img)
        
        if self.opt.lab:
            A_label =  A_path.split('/')[-1].split('_')[1]
            B_label = 82
            if self.opt.phase == 'test':
                A_label =  2
                B_label = 82
        else:
            A_label =  0
            B_label = 0
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,'A_label':A_label,'B_label':B_label,'C':C,'D':D} 

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
