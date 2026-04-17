import torch 
import numpy as np 
import random
import os
import nibabel as nib

#Test out segmentation loss with mean on a single slice example and see if it makes sense
def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

mse = torch.nn.MSELoss() #Criterion loss


hard_kernel_image = nib.load("/path/to/hard/kernel/image/slice").get_fdata()[:,:,0] 
soft_kernel_image = nib.load("/path/to/soft/kernel/image/slice").get_fdata()[:,:,0]
hard_kernel_mask = nib.load("/path/to/hard/kernel/mask/slice").get_fdata()[:,:,0]
soft_kernel_mask = nib.load("/path/to/soft/kernel/mask/slice").get_fdata()[:,:,0]

#Convert the numpy arrays to tensors
hard_kernel_image = torch.from_numpy(hard_kernel_image)
soft_kernel_image = torch.from_numpy(soft_kernel_image)
hard_kernel_mask = torch.from_numpy(hard_kernel_mask)
soft_kernel_mask = torch.from_numpy(soft_kernel_mask)

#Convert the tensors to float
hard_kernel_image = hard_kernel_image.unsqueeze(0).float().requires_grad_(True)
soft_kernel_image = soft_kernel_image.unsqueeze(0).float().requires_grad_(True)
hard_kernel_mask = hard_kernel_mask.unsqueeze(0).float()
soft_kernel_mask = soft_kernel_mask.unsqueeze(0).float()

#Compute the segmentation loss
real_labels_forward = torch.unique(hard_kernel_mask)
real_labels_forward = real_labels_forward[real_labels_forward != 0]
real_labels_backward = torch.unique(soft_kernel_mask) 
real_labels_backward = real_labels_backward[real_labels_backward != 0]

#Create a list to store the segmentation loss. The list should be the same size as the number of unique labels in the mask
mean_real_A = torch.zeros(len(real_labels_forward))
mean_real_B = torch.zeros(len(real_labels_backward)) 

#Approach1: Creating binary masks for each label and computing the mean of the image intensities for each label
for i, label in enumerate(real_labels_forward):
    bin_mask = (hard_kernel_mask == label).float()
    mean_real_A[i] = torch.mean(hard_kernel_image[bin_mask == 1])
    mean_real_B[i] = torch.mean(soft_kernel_image[bin_mask == 1])

print(mean_real_A)
print(mean_real_B)
loss = mse(mean_real_A, mean_real_B)
print(loss)

# #Approach 2: Not using the binary mask approach
# for i, label in enumerate(real_labels_forward):
#     mean_real_A[i] = torch.mean(hard_kernel_image[hard_kernel_mask == label])
#     mean_real_B[i] = torch.mean(soft_kernel_image[hard_kernel_mask == label])

# print(mean_real_A)
# print(mean_real_B)

# #Compute the mean squared error between the two lists
# loss = mse(mean_real_A, mean_real_B)
# print(loss)