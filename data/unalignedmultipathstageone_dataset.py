import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
import numpy as np 
import nibabel as nib 
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d
import pandas as pd 


#Must include circular masking strategy for B50f, B80f and B30f because the original images were used to slice the images, not the masks.
class UnalignedMultipathStageOneDataset(BaseDataset):
    """
    Dataset class for multipath kernel training in stage one. Takes either 4, 5 or 6 different kernels as inputs (Depending on memory).
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        #When training for four domains, use this code from lines 33-51
        df_path = "/lung_data/training_data/train_dataframes"

        self.b50f_df = pd.read_csv(os.path.join(df_path, 'df_b50f.csv'))
        self.b30f_df = pd.read_csv(os.path.join(df_path, 'df_b30f.csv'))
        self.bone_df = pd.read_csv(os.path.join(df_path, 'df_bone.csv'))
        self.lung_df = pd.read_csv(os.path.join(df_path, 'df_lung.csv'))
        self.b80f_df = pd.read_csv(os.path.join(df_path, 'df_b80f.csv'))
        self.standard_df = pd.read_csv(os.path.join(df_path, 'df_standard.csv'))

        self.b50f_image = sorted(self.b50f_df['image'].tolist())
        self.b50f_mask = sorted(self.b50f_df['mask'].tolist())
        self.b30f_image = sorted(self.b30f_df['image'].tolist())
        self.b30f_mask = sorted(self.b30f_df['mask'].tolist())
        self.bone_image = sorted(self.bone_df['image'].tolist())
        self.bone_mask = sorted(self.bone_df['mask'].tolist())
        self.lung_image = sorted(self.lung_df['image'].tolist())
        self.lung_mask = sorted(self.lung_df['mask'].tolist())
        self.b80f_image = sorted(self.b80f_df['image'].tolist())
        self.b80f_mask = sorted(self.b80f_df['mask'].tolist())
        self.std_image = sorted(self.standard_df['image'].tolist())
        self.std_mask = sorted(self.standard_df['mask'].tolist())

        self.b50f_size = len(self.b50f_image)
        self.b30f_size = len(self.b30f_image)
        self.bone_size = len(self.bone_image)
        self.lung_size = len(self.lung_image)
        self.b80f_size = len(self.b80f_image)
        self.std_size = len(self.std_image)

        self.max_length = 5000 # Maximum number of images to load from each domain for every epoch. Naive sampling approach to time one epoch. Need a more robust approach

        self.normalizer = interp1d([-1024, 3072], [-1,1])
    
    def shuffle_std_indices(self):
        self.std_indices = np.random.permutation(self.std_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        #STD domain has the most images, so we use it to index the other domains.
        #Create a permuation of the indexes for the standard domain. Randomly select an index for the standard domain. This is toensure that standard domain uses diffent indices for each epoch.

        # std_image_path = self.std_image[self.std_indices[index % self.std_size]]
        # std_mask_path = self.std_mask[self.std_indices[index % self.std_size]]
        #Randomize indexes for the other domains
        b50f_index = random.randint(0, self.b50f_size - 1)
        b30f_index = random.randint(0, self.b30f_size - 1)
        bone_index = random.randint(0, self.bone_size - 1)
        lung_index = random.randint(0, self.lung_size - 1)
        b80f_index = random.randint(0, self.b80f_size - 1)
        std_index = random.randint(0, self.std_size - 1)
       
        b50f_image_path = self.b50f_image[b50f_index]
        b50f_mask_path = self.b50f_mask[b50f_index]
        b30f_image_path = self.b30f_image[b30f_index]
        b30f_mask_path = self.b30f_mask[b30f_index]
        bone_image_path = self.bone_image[bone_index]
        bone_mask_path = self.bone_mask[bone_index]
        lung_image_path = self.lung_image[lung_index]
        lung_mask_path = self.lung_mask[lung_index]
        b80f_image_path = self.b80f_image[b80f_index]
        b80f_mask_path = self.b80f_mask[b80f_index]
        std_image_path = self.std_image[std_index]
        std_mask_path = self.std_mask[std_index]

        b50f_image, b50f_mask = self.mask_and_normalize(b50f_image_path, b50f_mask_path)
        b30f_image, b30f_mask = self.mask_and_normalize(b30f_image_path, b30f_mask_path)
        b80f_image, b80f_mask = self.mask_and_normalize(b80f_image_path, b80f_mask_path)
        bone_image, bone_mask = self.normalize(bone_image_path, bone_mask_path)
        lung_image, lung_mask = self.normalize(lung_image_path, lung_mask_path)
        std_image, std_mask = self.normalize(std_image_path, std_mask_path)

        return {
            'B50f_image': b50f_image, 'B50f_mask': b50f_mask, 'B30f_image': b30f_image, 'B30f_mask': b30f_mask,
            'BONE_image': bone_image, 'BONE_mask': bone_mask, 'LUNG_image': lung_image, 'LUNG_mask': lung_mask,
            'B80f_image': b80f_image, 'B80f_mask': b80f_mask, 'STANDARD_image': std_image, 'STANDARD_mask': std_mask,
            'B50f_image_path': b50f_image_path, 'B50f_mask_path': b50f_mask_path, 'B30f_image_path': b30f_image_path,
            'B30f_mask_path': b30f_mask_path, 'BONE_image_path': bone_image_path, 'BONE_mask_path': bone_mask_path,
            'LUNG_image_path': lung_image_path, 'LUNG_mask_path': lung_mask_path, 'B80f_image_path': b80f_image_path,
            'B80f_mask_path': b80f_mask_path, 'STANDARD_image_path': std_image_path, 'STANDARD_mask_path': std_mask_path
        }


    def __len__(self):
        """Return the total number of images in the dataset.

        """
        return self.max_length

    def mask_and_normalize(self, input_slice_path, mask_slice_path):
        image = nib.load(input_slice_path).get_fdata()[:,:,0]
        w, h = image.shape
        #Create a circular mask and apply to input image
        mask = np.zeros((w, h), dtype=image.dtype)
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y, w - center_x, h - center_y) 
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = dist_from_center <= radius
        image[~mask] = -1024

        clip_data = np.clip(image, -1024, 3072)
        norm_data = self.normalizer(clip_data)
        data_tensor = torch.from_numpy(norm_data)
        torch_data_tensor = data_tensor.unsqueeze(0).float()
        #Mask
        mask_data = nib.load(mask_slice_path).get_fdata()[:,:,0]
        mask_tensor = torch.from_numpy(mask_data)
        torch_mask_tensor = mask_tensor.unsqueeze(0).float()
        return torch_data_tensor, torch_mask_tensor
       

    def normalize(self, input_slice_path, mask_slice_path):
        #Data
        clip_data = np.clip(nib.load(input_slice_path).get_fdata()[:,:,0], -1024, 3072)
        norm_data = self.normalizer(clip_data)
        data_tensor = torch.from_numpy(norm_data)
        torch_data_tensor = data_tensor.unsqueeze(0).float()
        #Mask
        mask_data = nib.load(mask_slice_path).get_fdata()[:,:,0]
        mask_tensor = torch.from_numpy(mask_data)
        torch_mask_tensor = mask_tensor.unsqueeze(0).float()
        return torch_data_tensor, torch_mask_tensor