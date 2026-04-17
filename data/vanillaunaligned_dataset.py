import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np 
import nibabel as nib 
from scipy.interpolate import interp1d
import torch
from torch.utils.data import DataLoader, Dataset

class VanillaUnalignedDataset(BaseDataset):
    """
    Dataset class for mulitpath kernel conversion. Loads in nine kernels and returns the corresponding images.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        #When training for four domains, use this code from lines 33-51
        self.dir_A = os.path.join(opt.dataroot, opt.phase + opt.source_kernel)  
        self.dir_B = os.path.join(opt.dataroot, opt.phase + opt.target_kernel)
        self.mask_dirA = os.path.join(opt.dataroot, opt.source_kernel_mask)
        self.mask_dirB = os.path.join(opt.dataroot, opt.target_kernel_mask)
        print(self.dir_A)
        print(self.dir_B)
        print(self.mask_dirA)
        print(self.mask_dirB)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.mask_A_paths = sorted(make_dataset(self.mask_dirA, opt.max_dataset_size))
        self.mask_B_paths = sorted(make_dataset(self.mask_dirB, opt.max_dataset_size))    
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)    


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
        # Get the dataitems for 4 domains
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        mask_A_path = self.mask_A_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        mask_B_path = self.mask_B_paths[index_B]

        
        
        A, mask_A = self.return_data(A_path, mask_A_path)
        B, mask_B = self.return_data(B_path, mask_B_path)
    
        #Return a tuple of the kernel data instead of an indivodual kernel. (Needs to be implemented)
        return {'A': A, 'B': B, 'A_mask': mask_A, 'B_mask': mask_B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have different datasets with potentially different number of images,
        we take a maximum of all the datasets.
        """
        return max(self.A_size, self.B_size)
        # return max(self.subset_A, self.subset_B, self.subset_C, self.subset_D)


    def return_data(self, input_slice_path, mask_slice_path):
        nift_data = nib.load(input_slice_path).get_fdata()[:,:,0]
        tensor = torch.from_numpy(nift_data)
        torch_tensor = tensor.unsqueeze(0).float()
        mask_data = nib.load(mask_slice_path).get_fdata()[:,:,0]
        mask_tensor = torch.from_numpy(mask_data)
        mask_tensor = mask_tensor.unsqueeze(0).float()
        return torch_tensor, mask_tensor
