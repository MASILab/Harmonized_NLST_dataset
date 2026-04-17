import os 
import torch 
import numpy as np 
import nibabel as nib 
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from tqdm import tqdm
from glob import glob

#Formulate a dataloder for an individual nifti volume. Need to return the formulated volume as atensor or return a slice 
# from the volume as atensor. run the inference and build the volume for the ocnverted dataset. 
class InferenceDataloader(Dataset):
    def __init__(self, nii_path):
        self.nii_path = nii_path
        self.normalizer = interp1d([-1024, 3072], [-1,1])
        self.img = nib.load(self.nii_path)
        self.pix_data = None 

    def load_nii(self): #Handle some of the corrupt images (There are only 31 corrupt images)
        try:
            self.pix_data = self.img.get_fdata()
        except Exception as e:
            raise RuntimeError(f"Failed to read data from {self.nii_path}: {e}")


    def save_scan(self, scan_idx_slice_map, out_nii):
        new_scan = np.zeros(self.pix_data.shape, dtype=float)
        for slice_idx, slice_data in scan_idx_slice_map.items(): #slice_idx is the slice index and the slice_data is the slice that we need
            new_scan[:,:,slice_idx] = slice_data[:,:]
        
        new_scan = np.clip(new_scan, -1, 1) #image is normalized. Clip the images in the range of [-1,1] and then bring it back to HU.
        hu_normalizer = interp1d([-1,1], [-1024,3072])
        new_scan_hu = hu_normalizer(new_scan)
        nifti_img = nib.Nifti1Image(new_scan_hu, affine = self.img.affine, header = self.img.header)
        nib.save(nifti_img, out_nii)


    def __len__(self):
        return self.pix_data.shape[2]


    def __getitem__(self, slice_idx):
        nii_filename = os.path.basename(self.nii_path)
        case_name = nii_filename.replace('.nii.gz', '')

        input_data = np.zeros((1,512,512), dtype = float)
        input_data[0:,:,] = self.pix_data[:,:,slice_idx] #Grabbing a slice based on the index from the volume. Images are single channel and not three channel.

        input_data = np.clip(input_data, -1024, 3072)
        normalized = self.normalizer(input_data) #Image in the range of [-1,1] 

        return {'normalized_data':normalized, 
                'pid':case_name,
                'slice':slice_idx}