import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np 
import nibabel as nib 
from scipy.interpolate import interp1d
import torch


class UnalignedallkernelsDataset(BaseDataset):
    """
    Dataset class for mulitpath kernel conversion. Loads in nine kernels and returns the corresponding images.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_siemens_masked_hard')  #16343 images 
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_siemens_masked_soft')  #16343 images 
        self.dir_C = os.path.join(opt.dataroot, opt.phase + '_ge_bone_hard') 
        self.dir_D = os.path.join(opt.dataroot, opt.phase + '_ge_bone_soft') 
        self.dir_E = os.path.join(opt.dataroot, opt.phase + '_philips_hard')
        self.dir_F = os.path.join(opt.dataroot, opt.phase + '_philips_soft')
        self.dir_G = os.path.join(opt.dataroot, opt.phase + '_lung_hard')
        self.dir_H = os.path.join(opt.dataroot, opt.phase + '_lung_soft')
        #print the directories of the datasets.
        print(self.dir_A)
        print(self.dir_B)
        print(self.dir_C)
        print(self.dir_D)
        print(self.dir_E)
        print(self.dir_F)
        print(self.dir_G)
        print(self.dir_H)
        #Make datasets for all the kernels.
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))
        self.E_paths = sorted(make_dataset(self.dir_E, opt.max_dataset_size))
        self.F_paths = sorted(make_dataset(self.dir_F, opt.max_dataset_size))
        self.G_paths = sorted(make_dataset(self.dir_G, opt.max_dataset_size))
        self.H_paths = sorted(make_dataset(self.dir_H, opt.max_dataset_size))
        #Max lengths for all datasets
        self.A_size = len(self.A_paths) 
        self.B_size = len(self.B_paths)  
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        self.E_size = len(self.E_paths)
        self.F_size = len(self.F_paths)
        self.G_size = len(self.G_paths)
        self.H_size = len(self.H_paths)

        #Robust way of doing clipping 
        self.normalizer = interp1d([-1024,3072], [-1,1])

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        """
        #Use all the indexes but return only 20% of the max dataset in every epoch. 
        E_path = self.E_paths[index % self.E_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            index_C = index % self.C_size
            index_D = index % self.D_size
        else:   # randomize the index. 
            index_A = random.randint(0, self.A_size - 1)
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1)
            index_D = random.randint(0, self.D_size - 1)
            index_F = random.randint(0, self.F_size - 1)
            index_H = random.randint(0, self.H_size - 1)
            index_G = random.randint(0, self.G_size - 1)
        
        # print(f"Index A: {index_A}, Index B: {index_B}, Index C: {index_C}, Index D: {index_D}, Index E: {index_E}, Index F: {index_F}, Index G: {index % self.G_size}, Index H: {index_H}, Index I: {index_I}, Index J: {index_J}")
         
        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C]
        D_path = self.D_paths[index_D]
        F_path = self.F_paths[index_F]
        G_path = self.G_paths[index_G]
        H_path = self.H_paths[index_H]
        

        A = self.normalize(A_path)
        B = self.normalize(B_path)
        C = self.normalize(C_path)
        D = self.normalize(D_path) 
        E = self.normalize(E_path)
        F = self.normalize(F_path)
        G = self.normalize(G_path)
        H = self.normalize(H_path)

        #Return a tuple of the kernel data instead of an indivodual kernel. (Needs to be implemented)
        return {'A':A, 'B': B, 'C':C, 'D': D, 'E':E, 'F':F, 'G':G, 'H':H} 

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have different datasets with potentially different number of images,
        we take a maximum of all the datasets.
        """
        #Return 20% of the maximum dataset size. Each epoch covers 20% of the indices. In this way, the model should 
        # cover all the data in the entire training. 
        return int(0.2 * max(self.A_size, self.B_size, self.C_size, self.D_size, self.E_size, self.F_size, self.G_size, self.H_size))

    def normalize(self, input_slice_path):
        """Normalize input slice and return as a tensor ranging from [-1,1]"""
        nift_clip = np.clip(nib.load(input_slice_path).get_fdata()[:,:,0], -1024, 3072)
        norm = self.normalizer(nift_clip)
        tensor = torch.from_numpy(norm)
        torch_tensor = tensor.unsqueeze(0).float()
        return torch_tensor
