import torch
import os
from glob import glob
from tqdm import tqdm
from test_custom_dataloader import InferenceDataloader
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from models.networks import ResBlocklatent, ResNetEncoder, ResNetDecoder, G_decoder, G_encoder
from collections import OrderedDict
import torch.nn as nn
import pandas as pd
from joblib import Parallel, delayed

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class GenerateInferenceMultipathGAN:
    def __init__(self, model_checkpoint, inkernel, outkernel):
        self.model_checkpoint = model_checkpoint
        self.inkernel = inkernel #This will be a list of nifti files
        self.outkernel = outkernel #The output directory to have all the files that are harmonized

    def generate_images(self, enc, dec, failed_log="failed_images.log"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = torch.load(self.model_checkpoint)[enc]
        decoder = torch.load(self.model_checkpoint)[dec]
        encoderdict = OrderedDict()
        decoderdict = OrderedDict()
        for k, v in encoder.items():
            encoderdict["module." + k] = v
        for k, v in decoder.items():
            decoderdict["module." + k] = v

        shared_latent = ResBlocklatent(n_blocks=9, ngf=64, norm_layer=nn.InstanceNorm2d, padding_type='reflect')
        resencode = G_encoder(input_nc=1, ngf=64, netG_encoder="resnet_encoder", norm = 'instance', init_type='normal', init_gain=0.02, latent_layer=shared_latent, gpu_ids=[0])
        resdecode = G_decoder(output_nc=1, ngf=64, netG_decoder="resnet_decoder", norm = 'instance', init_type='normal', init_gain=0.02, gpu_ids=[0])
        resencode.load_state_dict(encoderdict)
        resdecode.load_state_dict(decoderdict)

        in_nii_path = self.inkernel
        out_nii = self.outkernel
        print(in_nii_path, out_nii)
        os.makedirs(out_nii, exist_ok=True)
        print(f'Identify {len(in_nii_path)} scans (.nii.gz)')

        #Set eval mode on and make sure that the gradients are off.
        with torch.no_grad():
            resencode.eval()
            resdecode.eval()
            for nii_path in tqdm(in_nii_path, total=len(in_nii_path)):
                try:
                    test_dataset = InferenceDataloader(nii_path)
                    test_dataset.load_nii()
                except Exception as e:
                    with open(failed_log, "a") as f:
                        f.write(f"{nii_path}, {str(e)}\n")
                    print(f"⚠️ Skipping corrupted file: {nii_path} ({e})")
                    continue  # skip to next file

                test_dataloader = DataLoader(dataset=test_dataset,
                                            batch_size=64, shuffle=False,
                                            num_workers=8, pin_memory=True)

                converted_scan_idx_slice_map = {}
                for i, data in enumerate(test_dataloader):
                    pid = data['pid']
                    norm_data = data['normalized_data'].float().to(device)
                    latent = resencode(norm_data)
                    fake_image = resdecode(latent)
                    fake_image_numpy = fake_image.cpu().numpy()
                    slice_idx_list = data['slice'].data.cpu().numpy().tolist()
                    for idx, slice_index in enumerate(slice_idx_list):
                        converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :]

                nii_file_name = os.path.basename(nii_path)
                converted_image = os.path.join(out_nii, nii_file_name)
                test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
                print(f"{nii_file_name} converted!")
    
 

def run_harmonization_NLST_paired_kernels_T0():
    config_inference_stage1 = {
        'stage1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/152_net_gendisc_weights.pth",
        'stage2_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/56_net_gendisc_weights.pth", #Temp epoch, have not run validation on the other model which needs to be run
        'B50ftoB30f': "/NLST_harmonized_images/paired_kernels_harmonized/T0/B50ftoB30f/harmonized_B50f_training_data", #Already harmonized, needs to be symlinked
        "B80ftoB30f": "/NLST_harmonized_images/paired_kernels_harmonized/T0/B80ftoB30f/harmonized_B80f_training_data", #Already harmonized, needs to be symlinked to this directory
        "BONEtoSTANDARD": "/NLST_harmonized_images/paired_kernels_harmonized/T0/BONEtoSTANDARD/harmonized_BONE_training_data",
        "LUNGtoSTANDARD": "/NLST_harmonized_images/paired_kernels_harmonized/T0/LUNGtoSTANDARD/harmonized_LUNG_training_data",
        "DtoB": "/NLST_harmonized_images/paired_kernels_harmonized/T0/DtoB/harmonized_D_training_data",
        "DtoC": "/NLST_harmonized_images/paired_kernels_harmonized/T0/DtoC/harmonized_D_training_data",
    }
     
    df_training_data = pd.read_csv("NLST_t0_training_data_multipathcycleGAN_6_9_25.csv")
    df_training_data = df_training_data[df_training_data['split'] != 'train'] #Only validation and testing data but I think I need to run on training data as well. Ran this for my paper 


    b50f_files = df_training_data[(df_training_data["kernel_pair"] == "B30f_B50f") & (df_training_data['Kernel'] == "B50f")]['File_path_for_TotalSegmentator'].to_list()
    b80f_files = df_training_data[(df_training_data["kernel_pair"] == "B30f_B80f") & (df_training_data['Kernel'] == "B80f")]['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_training_data[(df_training_data["kernel_pair"] == "STANDARD_BONE") & (df_training_data['Kernel'] == "BONE")]['File_path_for_TotalSegmentator'].to_list()
    lung_files = df_training_data[(df_training_data["kernel_pair"] == "STANDARD_LUNG") & (df_training_data['Kernel'] == "LUNG")]['File_path_for_TotalSegmentator'].to_list()
    d_c_files = df_training_data[(df_training_data["kernel_pair"] == "C_D") & (df_training_data['Kernel'] == "D")]['File_path_for_TotalSegmentator'].to_list()
    d_b_files = df_training_data[(df_training_data["kernel_pair"] == "B_D") & (df_training_data['Kernel'] == "D")]['File_path_for_TotalSegmentator'].to_list()

    print("Length of B50f files: ", len(b50f_files))
    print("Length of B80f files: ", len(b80f_files))
    print("Length of BONE files: ", len(bone_files))
    print("Length of LUNG files: ", len(lung_files))
    print("Length of D to C files: ", len(d_c_files))
    print("Length of D to B files: ", len(d_b_files))

    b50ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                               inkernel=b50f_files, 
                                               outkernel=config_inference_stage1['B50ftoB30f'])

    b80ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                               inkernel=b80f_files, 
                                               outkernel=config_inference_stage1['B80ftoB30f'])

    bonetostandard = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                               inkernel=bone_files, 
                                               outkernel=config_inference_stage1['BONEtoSTANDARD'])

    lungtostandard = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                               inkernel=lung_files, 
                                               outkernel=config_inference_stage1['LUNGtoSTANDARD'])

    dtob = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint'], 
                                               inkernel=d_b_files, 
                                               outkernel=config_inference_stage1['DtoB'])

    dtoc = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint'], 
                                               inkernel=d_c_files, 
                                               outkernel=config_inference_stage1['DtoC'])


    dtob.generate_images("G_D_encoder", "G_B_decoder") # GPU0
    b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder")#GPU0

    lungtostandard.generate_images("G_LUNG_encoder", "G_STDD_decoder") #GPU 0
    dtoc.generate_images("G_D_encoder", "G_C_decoder") #GPU 0

    b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder") # GPU 1
    bonetostandard.generate_images("G_BONE_encoder", "G_STD_decoder") #GPU 1



    
    

def run_harmonization_NLST_paired_kernels_T1():
    config_inference_stage = {
        'stage1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/152_net_gendisc_weights.pth",
        'stage2_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/56_net_gendisc_weights.pth", 
        'B50ftoB30f': "/NLST_harmonized_images/paired_kernels_harmonized/T1/B50ftoB30f/harmonized_B50f", 
        "B80ftoB30f": "/NLST_harmonized_images/paired_kernels_harmonized/T1/B80ftoB30f/harmonized_B80f", 
        "BONEtoSTANDARD": "/NLST_harmonized_images/paired_kernels_harmonized/T1/BONEtoSTANDARD/harmonized_BONE",
        "LUNGtoSTANDARD": "/NLST_harmonized_images/paired_kernels_harmonized/T1/LUNGtoSTANDARD/harmonized_LUNG",
        "DtoB": "/NLST_harmonized_images/paired_kernels_harmonized/T1/DtoB/harmonized_D",
        "DtoC": "/NLST_harmonized_images/paired_kernels_harmonized/T1/DtoC/harmonized_D",
    }

    df_t1 = pd.read_csv("/NLST_T1_all_resampled_pairs_aligned_post_QA_symlink_correction_resampling_10_16_25.csv")

    #HArmonize to the corresponding soft kernel
    b50f_files = df_t1[(df_t1["kernel_pair"] == "B30f_B50f") & (df_t1['Kernel'] == "B50f")]['File_path_for_TotalSegmentator'].to_list()
    b80f_files = df_t1[(df_t1["kernel_pair"] == "B30f_B80f") & (df_t1['Kernel'] == "B80f")]['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_t1[(df_t1["kernel_pair"] == "STANDARD_BONE") & (df_t1['Kernel'] == "BONE")]['File_path_for_TotalSegmentator'].to_list()
    lung_files = df_t1[(df_t1["kernel_pair"] == "STANDARD_LUNG") & (df_t1['Kernel'] == "LUNG")]['File_path_for_TotalSegmentator'].to_list()
    d_c_files = df_t1[(df_t1["kernel_pair"] == "C_D") & (df_t1['Kernel'] == "D")]['File_path_for_TotalSegmentator'].to_list()
    d_b_files = df_t1[(df_t1["kernel_pair"] == "B_D") & (df_t1['Kernel'] == "D")]['File_path_for_TotalSegmentator'].to_list()  

    print("Length of B50f files: ", len(b50f_files))
    print("Length of B80f files: ", len(b80f_files))
    print("Length of BONE files: ", len(bone_files))
    print("Length of LUNG files: ", len(lung_files))
    print("Length of D to C files: ", len(d_c_files))
    print("Length of D to B files: ", len(d_b_files)) 


    b50ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'], 
                                               inkernel=b50f_files, 
                                               outkernel=config_inference_stage['B50ftoB30f'])
    
    b80ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'], 
                                               inkernel=b80f_files, 
                                               outkernel=config_inference_stage['B80ftoB30f'])
    
    bonetostandard = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'], 
                                               inkernel=bone_files, 
                                               outkernel=config_inference_stage['BONEtoSTANDARD'])
    
    lungtostandard = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'],
                                                  inkernel=lung_files, 
                                                  outkernel=config_inference_stage['LUNGtoSTANDARD'])
    
    dtob = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage2_checkpoint'], 
                                               inkernel=d_b_files, 
                                               outkernel=config_inference_stage['DtoB'])
    
    dtoc = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage2_checkpoint'], 
                                               inkernel=d_c_files, 
                                               outkernel=config_inference_stage['DtoC'])
    

    dtob.generate_images("G_D_encoder", "G_B_decoder") # GPU0
    b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder")#GPU0

    lungtostandard.generate_images("G_LUNG_encoder", "G_STD_decoder") #GPU 1
    dtoc.generate_images("G_D_encoder", "G_C_decoder") #GPU 1

    b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder") # GPU 2
    bonetostandard.generate_images("G_BONE_encoder", "G_STD_decoder") #GPU 2


def run_harmonization_NLST_paired_kernels_T2():
    config_inference_stage = {
        'stage1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/152_net_gendisc_weights.pth",
        'stage2_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/56_net_gendisc_weights.pth", 
        'B50ftoB30f': "/NLST_harmonized_images/paired_kernels_harmonized/T2/B50ftoB30f/harmonized_B50f", 
        "B80ftoB30f": "/NLST_harmonized_images/paired_kernels_harmonized/T2/B80ftoB30f/harmonized_B80f", 
        "BONEtoSTANDARD": "/NLST_harmonized_images/paired_kernels_harmonized/T2/BONEtoSTANDARD/harmonized_BONE",
        "LUNGtoSTANDARD": "/NLST_harmonized_images/paired_kernels_harmonized/T2/LUNGtoSTANDARD/harmonized_LUNG",
        "DtoB": "/NLST_harmonized_images/paired_kernels_harmonized/T2/DtoB/harmonized_D",
        "DtoC": "/NLST_harmonized_images/paired_kernels_harmonized/T2/DtoC/harmonized_D",
    }
      
    df_t2 = pd.read_csv("/NLST_T2_all_resampled_pairs_aligned_post_QA_symlink_correction_resampling_10_16_25.csv")  

    #HArmonize to the corresponding soft kernel
    b50f_files = df_t2[(df_t2["kernel_pair"] == "B30f_B50f") & (df_t2['Kernel'] == "B50f")]['File_path_for_TotalSegmentator'].to_list()
    b80f_files = df_t2[(df_t2["kernel_pair"] == "B30f_B80f") & (df_t2['Kernel'] == "B80f")]['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_t2[(df_t2["kernel_pair"] == "STANDARD_BONE") & (df_t2['Kernel'] == "BONE")]['File_path_for_TotalSegmentator'].to_list()
    lung_files = df_t2[(df_t2["kernel_pair"] == "STANDARD_LUNG") & (df_t2['Kernel'] == "LUNG")]['File_path_for_TotalSegmentator'].to_list()
    d_c_files = df_t2[(df_t2["kernel_pair"] == "C_D") & (df_t2['Kernel'] == "D")]['File_path_for_TotalSegmentator'].to_list()
    d_b_files = df_t2[(df_t2["kernel_pair"] == "B_D") & (df_t2['Kernel'] == "D")]['File_path_for_TotalSegmentator'].to_list()

    print("Length of B50f files: ", len(b50f_files))
    print("Length of B80f files: ", len(b80f_files))
    print("Length of BONE files: ", len(bone_files))        
    print("Length of LUNG files: ", len(lung_files))
    print("Length of D to C files: ", len(d_c_files))
    print("Length of D to B files: ", len(d_b_files))

    b50ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'], 
                                               inkernel=b50f_files, 
                                               outkernel=config_inference_stage['B50ftoB30f'])  
    
    b80ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'], 
                                               inkernel=b80f_files, 
                                               outkernel=config_inference_stage['B80ftoB30f'])

    bonetostandard = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'], 
                                               inkernel=bone_files, 
                                               outkernel=config_inference_stage['BONEtoSTANDARD'])
    
    lungtostandard = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage1_checkpoint'],
                                                    inkernel=lung_files,
                                                    outkernel=config_inference_stage['LUNGtoSTANDARD'])
    
    dtob = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage2_checkpoint'],
                                                  inkernel=d_b_files, 
                                                  outkernel=config_inference_stage['DtoB'])
    
    dtoc = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage['stage2_checkpoint'],
                                                    inkernel=d_c_files,
                                                    outkernel=config_inference_stage['DtoC'])
    
    dtob.generate_images("G_D_encoder", "G_B_decoder") # GPU0
    b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder")#GPU0

    lungtostandard.generate_images("G_LUNG_encoder", "G_STD_decoder") #GPU 1
    dtoc.generate_images("G_D_encoder", "G_C_decoder") #GPU 1

    b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder") # GPU 2
    bonetostandard.generate_images("G_BONE_encoder", "G_STD_decoder") #GPU 2


if __name__ == "__main__":
    # run_harmonization_NLST_paired_kernels_T0()
    # run_harmonization_NLST_paired_kernels_T1()
    run_harmonization_NLST_paired_kernels_T2()