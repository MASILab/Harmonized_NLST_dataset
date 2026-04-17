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


# #Config dictionary with image paths, out paths and checkpoint file
# #Stage 1: Harmonize all images to Siemens B30f (soft kernel) and Siemens B50f (hard kernel). Currently, we pick an epoch based on our previous publication
# #Kernels are B50f, B80f, BONE, LUNG, STANDARD and B30f in Stage 1. 
# # Need to process the images in batches and ensure that things make sense. 
def run_harmonization_NLST_to_B30f():
    config_inference_stage1 = {
        'stage1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/152_net_gendisc_weights.pth",
        'stage2_checkpoint_temp': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/56_net_gendisc_weights.pth", #Temp epoch, have not run validation on the other model which needs to be run
        'b50ftob30f': "/harmonized_to_B30f/B50ftoB30f",
        'b80ftob30f': "/harmonized_to_B30f/B80ftoB30f",
        'bonetob30f': "/harmonized_to_B30f/BONEttoB30f",
        'lungtob30f': "//harmonized_to_B30f/LUNGtoB30f",
        'btob30f': '/harmonized_to_B30f/BtoB30f_temp', 
        'ctob30f': '/harmonized_to_B30f/CtoB30f_temp',
        'dtob30f': '/harmonized_to_B30f/DtoB30f_temp',
        'stdtob30f': '/harmonized_to_B30f/STANDARDtoB30f'
    }

    #Use the paired dataframe and run things on T0
    df = pd.read_csv("/NLST_T0_paired_resampled_file_paths_6_8_25_fixed_symlinks_reoriented_final_spreadsheet_resampled_img_dims_circular_masked_with_totalsegmentator_paths_reoriented.csv")
    df_remaining = pd.read_csv("/8_17_25_NLST_T0_remaining_data_fixed_symlinks_resmapled_masked_latest.csv")

    #stage 1 and 2 paired data
    b80f_files = df[df['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list() #Completed
    b50f_files = df[df['Kernel'] == 'B50f']['File_path_for_TotalSegmentator'].to_list() #Completed 
    bone_files = df[df['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list() #Completed
    lung_files = df[df['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list() #Completed
    std_files = df[df['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list() #Have to run later since there are a lot of files
    b_files = df[df['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files = df[df['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files = df[df['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list()

    #Stage 1 and 2 remaining data 
    b80f_files_remaining = df_remaining[df_remaining['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list()
    b50f_files_remaining = df_remaining[df_remaining['Kernel'] == 'B50f']['File_path_for_TotalSegmentator'].to_list()
    bone_files_remaining = df_remaining[df_remaining['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list()
    lung_files_remaining = df_remaining[df_remaining['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list()
    std_files_remaining = df_remaining[df_remaining['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list() #Have to run later since there are a lot of files
    b_files_remaining = df_remaining[df_remaining['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files_remaining = df_remaining[df_remaining['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files_remaining = df_remaining[df_remaining['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list()

    #Some images from STANDARD are yet to run,  

    config_stdtob30f = os.listdir(config_inference_stage1['stdtob30f']) 

    #Find the files in std_files_remianing that are not in config_stdtob30f
    std_files_remaining_batch = [file for file in std_files_remaining if os.path.basename(file) not in config_stdtob30f]
    print("Files remaining for harmonization in STANDARD kernel: ", len(std_files_remaining_batch))


    b80ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                                inkernel=b80f_files, 
                                                outkernel=config_inference_stage1['b80ftob30f'])

    b50ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                                inkernel=b50f_files, 
                                                outkernel=config_inference_stage1['b50ftob30f'])

    bonetob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                                inkernel=bone_files, 
                                                outkernel=config_inference_stage1['bonetob30f'])

    lungtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'], 
                                                inkernel=lung_files, 
                                                outkernel=config_inference_stage1['lungtob30f']) 
    
    stdtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'],
                                                inkernel=std_files,
                                                outkernel=config_inference_stage1['stdtob30f']) #Have to run
    
    btob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint_temp'],
                                                inkernel=b_files,
                                                outkernel=config_inference_stage1['btob30f'])
    
    ctob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint_temp'],
                                                inkernel=c_files,
                                                outkernel=config_inference_stage1['ctob30f'])
    
    dtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint_temp'],
                                                inkernel=d_files,
                                                outkernel=config_inference_stage1['dtob30f'])
    
    
    b80ftob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'],
                                                inkernel=b80f_files_remaining,
                                                outkernel=config_inference_stage1['b80ftob30f'])
    
    b50ftob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'],
                                                inkernel=b50f_files_remaining,
                                                outkernel=config_inference_stage1['b50ftob30f'])
    
    bonetob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'],
                                                inkernel=bone_files_remaining,
                                                outkernel=config_inference_stage1['bonetob30f'])
    
    lungtob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'],
                                                inkernel=lung_files_remaining,
                                                outkernel=config_inference_stage1['lungtob30f'])

    stdtob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'],
                                                inkernel=std_files_remaining,
                                                outkernel=config_inference_stage1['stdtob30f'])

    btob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint_temp'],
                                                inkernel=b_files_remaining,
                                                outkernel=config_inference_stage1['btob30f'])
    
    ctob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint_temp'],
                                                inkernel=c_files_remaining,
                                                outkernel=config_inference_stage1['ctob30f'])
    
    dtob30f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage2_checkpoint_temp'],
                                                inkernel=d_files_remaining,
                                                outkernel=config_inference_stage1['dtob30f'])

    stdtob30f_remaining_batch = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_stage1['stage1_checkpoint'],
                                                inkernel=std_files_remaining_batch,
                                                outkernel=config_inference_stage1['stdtob30f'])
                                                              
    
    
    
    b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder") #completed
    b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder") #completed
    bonetob30f.generate_images("G_BONE_encoder", "G_B30f_decoder") #completed
    lungtob30f.generate_images("G_LUNG_encoder", "G_B30f_decoder") #completed
    btob30f.generate_images("G_B_encoder", "G_B30f_decoder")#completed
    ctob30f.generate_images("G_C_encoder", "G_B30f_decoder")#completed
    dtob30f.generate_images("G_D_encoder", "G_B30f_decoder")#completed
    stdtob30f.generate_images("G_STD_encoder", "G_B30f_decoder")#completed


    b80ftob30f_remaining.generate_images("G_B80f_encoder", "G_B30f_decoder")#completed
    b50ftob30f_remaining.generate_images("G_B50f_encoder", "G_B30f_decoder")#completed
    bonetob30f_remaining.generate_images("G_BONE_encoder", "G_B30f_decoder")#completed
    lungtob30f_remaining.generate_images("G_LUNG_encoder", "G_B30f_decoder")#completed
    btob30f_remaining.generate_images("G_B_encoder", "G_B30f_decoder")#completed
    ctob30f_remaining.generate_images("G_C_encoder", "G_B30f_decoder")#completed
    dtob30f_remaining.generate_images("G_D_encoder", "G_B30f_decoder")#completed
    stdtob30f_remaining.generate_images("G_STD_encoder", "G_B30f_decoder")#completed

    stdtob30f_remaining_batch.generate_images("G_STD_encoder", "G_B30f_decoder")#completed



def run_harmonization_NLST_to_B50f():
    config_inferece_b50f = {
        'stage_1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/198_net_gendisc_weights.pth",
        'stage_2_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/190_net_gendisc_weights.pth", 
        'b30ftob50f': "/harmonized_to_B50f/B30ftoB50f",  
        'b80ftob50f': "/harmonized_to_B50f/B80ftoB50f", 
        'bonetob50f': "/harmonized_to_B50f/BONEtoB50f", 
        'lungtob50f': "/harmonized_to_B50f/LUNGtoB50f", 
        'stdtob50f': "/harmonized_to_B50f/STANDARDtoB50f",
        'btob50f': "/harmonized_to_B50f/BtoB50f",
        'ctob50f': "/harmonized_to_B50f/CtoB50f",
        'dtob50f': "/harmonized_to_B50f/DtoB50f"
    }

    df = pd.read_csv("/NLST_T0_paired_resampled_file_paths_6_8_25_fixed_symlinks_reoriented_final_spreadsheet_resampled_img_dims_circular_masked_with_totalsegmentator_paths_reoriented.csv")
    df_remaining = pd.read_csv("/8_17_25_NLST_T0_remaining_data_fixed_symlinks_resmapled_masked_latest.csv")

    b80f_files = df[df['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list() #Completed
    b30f_files = df[df['Kernel'] == 'B30f']['File_path_for_TotalSegmentator'].to_list() #Completed 
    bone_files = df[df['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list() #Completed
    lung_files = df[df['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list() #Completed
    std_files = df[df['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list() #Have to run later since there are a lot of files
    b_files = df[df['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files = df[df['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files = df[df['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list()

    #Stage 1 and 2 remaining data 
    b80f_files_remaining = df_remaining[df_remaining['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list()
    b30f_files_remaining = df_remaining[df_remaining['Kernel'] == 'B30f']['File_path_for_TotalSegmentator'].to_list()
    bone_files_remaining = df_remaining[df_remaining['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list()
    lung_files_remaining = df_remaining[df_remaining['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list()
    std_files_remaining = df_remaining[df_remaining['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list() #Have to run later since there are a lot of files
    b_files_remaining = df_remaining[df_remaining['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files_remaining = df_remaining[df_remaining['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files_remaining = df_remaining[df_remaining['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list()


    btob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=b_files,
                                            outkernel=config_inferece_b50f['btob50f'])
    
    btob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=b_files_remaining,
                                            outkernel=config_inferece_b50f['btob50f'])
    
    ctob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=c_files,
                                            outkernel=config_inferece_b50f['ctob50f'])
    
    ctob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=c_files_remaining,
                                            outkernel=config_inferece_b50f['ctob50f'])
    
    dtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=d_files,
                                            outkernel=config_inferece_b50f['dtob50f'])
    
    dtob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                                    inkernel=d_files_remaining,
                                                    outkernel=config_inferece_b50f['dtob50f'])

    #Stage1 
    b30ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=b30f_files,
                                               outkernel=config_inferece_b50f['b30ftob50f'])
    
    b30ftob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                                         inkernel=b30f_files_remaining,
                                                         outkernel=config_inferece_b50f['b30ftob50f'])
    
    b80ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=b80f_files,
                                               outkernel=config_inferece_b50f['b80ftob50f'])

    b80ftob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                                         inkernel=b80f_files_remaining,
                                                         outkernel=config_inferece_b50f['b80ftob50f'])

    bonetob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=bone_files,
                                               outkernel=config_inferece_b50f['bonetob50f'])

    bonetob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                                       inkernel=bone_files_remaining,
                                                       outkernel=config_inferece_b50f['bonetob50f'])

    stdtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=std_files,
                                               outkernel=config_inferece_b50f['stdtob50f'])
    
    stdtob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                                        inkernel=std_files_remaining,
                                                        outkernel=config_inferece_b50f['stdtob50f'])
    
    lungtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                                  inkernel=lung_files,
                                                    outkernel=config_inferece_b50f['lungtob50f'])

    lungtob50f_remaining = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                                        inkernel=lung_files_remaining,
                                                        outkernel=config_inferece_b50f['lungtob50f'])
    


    # I ch


#Longitudinal data for T1 and T2 
def run_harmonization_NLST_T1_to_B30f():
    config_inference_b30f_t1 = {
        'stage1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/152_net_gendisc_weights.pth",
        'stage2_checkpoint_temp': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/56_net_gendisc_weights.pth", #Temp epoch, have not run validation on the other model which needs to be run
        'b50ftob30f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/B50ftoB30f",
        'b80ftob30f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/B80ftoB30f",
        'bonetob30f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/BONEtoB30f",
        'lungtob30f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/LUNGtoB30f",
        'btob30f': '/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/BtoB30f',
        'ctob30f': '/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/CtoB30f',
        'dtob30f': '/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/DtoB30f',
        'stdtob30f': '/NLST_harmonized_images/T1_harmonized/harmonized_to_B30f/STANDARDtoB30f'
    }

    df_t1 = pd.read_csv("/spreadsheet_with_resampled_paths_post_QA/NLST_T1_all_resampled_paths_post_QA_symlink_corrected_resampled_masked_10_4_25.csv")

    b80f_files = df_t1[df_t1['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list()
    b50f_files = df_t1[df_t1['Kernel'] == 'B50f']['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_t1[df_t1['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list()   
    lung_files = df_t1[df_t1['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list()
    std_files = df_t1[df_t1['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list()
    b_files = df_t1[df_t1['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files = df_t1[df_t1['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files = df_t1[df_t1['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list() 
    
    print("Number of files in T1: ", len(df_t1))

    b80ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage1_checkpoint'],
                                                inkernel=b80f_files,
                                                outkernel=config_inference_b30f_t1['b80ftob30f'])
    
    b50ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage1_checkpoint'],
                                                inkernel=b50f_files,
                                                outkernel=config_inference_b30f_t1['b50ftob30f'])
    
    bonetob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage1_checkpoint'],
                                                inkernel=bone_files,
                                                outkernel=config_inference_b30f_t1['bonetob30f'])

    lungtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage1_checkpoint'],
                                                inkernel=lung_files,
                                                outkernel=config_inference_b30f_t1['lungtob30f'])
    
    stdtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage1_checkpoint'],
                                                inkernel=std_files,
                                                outkernel=config_inference_b30f_t1['stdtob30f']) #
    
    btob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage2_checkpoint_temp'],
                                                inkernel=b_files,
                                                outkernel=config_inference_b30f_t1['btob30f'])
    
    ctob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage2_checkpoint_temp'],
                                                inkernel=c_files,
                                                outkernel=config_inference_b30f_t1['ctob30f'])
    
    dtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t1['stage2_checkpoint_temp'],
                                                inkernel=d_files,
                                                outkernel=config_inference_b30f_t1['dtob30f'])  

    b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder") 
    btob30f.generate_images("G_B_encoder", "G_B30f_decoder")
    #b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder") 
    #bonetob30f.generate_images("G_BONE_encoder", "G_B30f_decoder") 
    dtob30f.generate_images("G_D_encoder", "G_B30f_decoder")
    lungtob30f.generate_images("G_LUNG_encoder", "G_B30f_decoder") 
    ctob30f.generate_images("G_C_encoder", "G_B30f_decoder")
    
    #stdtob30f.generate_images("G_STD_encoder", "G_B30f_decoder")


def run_harmonization_NLST_T1_to_B50f():

    config_inference_b50f_t1 = {
        'stage_1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/198_net_gendisc_weights.pth",
        'stage_2_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/190_net_gendisc_weights.pth", 
        'b30ftob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/B30ftoB50f",  
        'b80ftob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/B80ftoB50f", 
        'bonetob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/BONEtoB50f", 
        'lungtob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/LUNGtoB50f", 
        'stdtob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/STANDARDtoB50f",
        'btob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/BtoB50f",
        'ctob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/CtoB50f",
        'dtob50f': "/NLST_harmonized_images/T1_harmonized/harmonized_to_B50f/DtoB50f"
    }   

    df_t1 = pd.read_csv("/spreadsheet_with_resampled_paths_post_QA/NLST_T1_all_resampled_paths_post_QA_symlink_corrected_resampled_masked_10_4_25.csv")
    print("Number of files in T1: ", len(df_t1))

    b80f_files = df_t1[df_t1['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list()
    b30f_files = df_t1[df_t1['Kernel'] == 'B30f']['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_t1[df_t1['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list()   
    lung_files = df_t1[df_t1['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list()
    std_files = df_t1[df_t1['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list()
    b_files = df_t1[df_t1['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files = df_t1[df_t1['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files = df_t1[df_t1['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list()

    btob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_2_checkpoint'],
                                            inkernel=b_files,
                                            outkernel=config_inference_b50f_t1['btob50f'])

    ctob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_2_checkpoint'],
                                            inkernel=c_files,
                                            outkernel=config_inference_b50f_t1['ctob50f'])

    dtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_2_checkpoint'],
                                            inkernel=d_files,
                                            outkernel=config_inference_b50f_t1['dtob50f'])
    
    b30ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_1_checkpoint'],
                                               inkernel=b30f_files,
                                               outkernel=config_inference_b50f_t1['b30ftob50f'])
    
    b80ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_1_checkpoint'],
                                               inkernel=b80f_files,
                                               outkernel=config_inference_b50f_t1['b80ftob50f'])
    
    bonetob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_1_checkpoint'],
                                                inkernel=bone_files,  
                                                outkernel=config_inference_b50f_t1['bonetob50f'])
    
    lungtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_1_checkpoint'],
                                               inkernel=lung_files,
                                               outkernel=config_inference_b50f_t1['lungtob50f'])
    
    stdtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t1['stage_1_checkpoint'],
                                                inkernel=std_files,
                                                outkernel=config_inference_b50f_t1['stdtob50f']) 
    
    b80ftob50f.generate_images("G_B80f_encoder", "G_B50f_decoder")
    btob50f.generate_images("G_B_encoder", "G_B50f_decoder")
    dtob50f.generate_images("G_D_encoder", "G_B50f_decoder")
    lungtob50f.generate_images("G_LUNG_encoder", "G_B50f_decoder")
    ctob50f.generate_images("G_C_encoder", "G_B50f_decoder")
    #b30ftob50f.generate_images("G_B30f_encoder", "G_B50f_decoder")
    #bonetob50f.generate_images("G_BONE_encoder", "G_B50f_decoder")
    #stdtob50f.generate_images("G_STD_encoder", "G_B50f_decoder")


def run_harmonization_NLST_T2_to_B30f():

    config_inference_b30f_t2 = {
        'stage1_checkpoint': "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/152_net_gendisc_weights.pth",
        'stage2_checkpoint_temp': "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/56_net_gendisc_weights.pth", #Temp epoch, have not run validation on the other model which needs to be run
        'b50ftob30f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/B50ftoB30f",
        'b80ftob30f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/B80ftoB30f",
        'bonetob30f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/BONEtoB30f",
        'lungtob30f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/LUNGtoB30f",
        'btob30f': '/NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/BtoB30f',
        'ctob30f': '/NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/CtoB30f',
        'dtob30f': '//NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/DtoB30f',
        'stdtob30f': '/NLST_harmonized_images/T2_harmonized/harmonized_to_B30f/STANDARDtoB30f'
    }

    df_t2 = pd.read_csv("/spreadsheet_with_resampled_paths_post_QA/NLST_T2_all_resampled_paths_post_QA_symlink_corrected_resampled_masked_10_4_25.csv")

    b80f_files = df_t2[df_t2['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list()
    b50f_files = df_t2[df_t2['Kernel'] == 'B50f']['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_t2[df_t2['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list()
    lung_files = df_t2[df_t2['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list()
    std_files = df_t2[df_t2['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list()
    b_files = df_t2[df_t2['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files = df_t2[df_t2['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files = df_t2[df_t2['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list()

    print("Number of files in T1: ", len(df_t2))

    b80ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage1_checkpoint'],
                                                inkernel=b80f_files,
                                                outkernel=config_inference_b30f_t2['b80ftob30f'])
    
    b50ftob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage1_checkpoint'],
                                                inkernel=b50f_files,
                                                outkernel=config_inference_b30f_t2['b50ftob30f'])
    
    bonetob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage1_checkpoint'],
                                                inkernel=bone_files,
                                                outkernel=config_inference_b30f_t2['bonetob30f'])

    lungtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage1_checkpoint'],
                                                inkernel=lung_files,
                                                outkernel=config_inference_b30f_t2['lungtob30f'])

    stdtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage1_checkpoint'],
                                                inkernel=std_files,
                                                outkernel=config_inference_b30f_t2['stdtob30f'])

    btob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage2_checkpoint_temp'],
                                                inkernel=b_files,
                                                outkernel=config_inference_b30f_t2['btob30f'])

    ctob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage2_checkpoint_temp'],
                                                inkernel=c_files,
                                                outkernel=config_inference_b30f_t2['ctob30f'])

    dtob30f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b30f_t2['stage2_checkpoint_temp'],
                                                inkernel=d_files,
                                                outkernel=config_inference_b30f_t2['dtob30f'])

    b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder") 
    btob30f.generate_images("G_B_encoder", "G_B30f_decoder")
    dtob30f.generate_images("G_D_encoder", "G_B30f_decoder")
    lungtob30f.generate_images("G_LUNG_encoder", "G_B30f_decoder")
    ctob30f.generate_images("G_C_encoder", "G_B30f_decoder") 
    # b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder") 
    # bonetob30f.generate_images("G_BONE_encoder", "G_B30f_decoder") 
    # stdtob30f.generate_images("G_STD_encoder", "G_B30f_decoder")

def run_harmonization_NLST_T2_to_B50f():

    config_inference_b50f_t2 = {
        'stage_1_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/198_net_gendisc_weights.pth",
        'stage_2_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/190_net_gendisc_weights.pth", 
        'b30ftob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/B30ftoB50f",  
        'b80ftob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/B80ftoB50f", 
        'bonetob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/BONEtoB50f", 
        'lungtob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/LUNGtoB50f", 
        'stdtob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/STANDARDtoB50f",
        'btob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/BtoB50f",
        'ctob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/CtoB50f",
        'dtob50f': "/NLST_harmonized_images/T2_harmonized/harmonized_to_B50f/DtoB50f"
    }

    df_t2 = pd.read_csv("/NLST_T2_all_resampled_paths_post_QA_symlink_corrected_resampled_masked_10_4_25.csv")

    print("Number of files in T1: ", len(df_t2))

    b80f_files = df_t2[df_t2['Kernel'] == 'B80f']['File_path_for_TotalSegmentator'].to_list()
    b30f_files = df_t2[df_t2['Kernel'] == 'B30f']['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_t2[df_t2['Kernel'] == 'BONE']['File_path_for_TotalSegmentator'].to_list()   
    lung_files = df_t2[df_t2['Kernel'] == 'LUNG']['File_path_for_TotalSegmentator'].to_list()
    std_files = df_t2[df_t2['Kernel'] == 'STANDARD']['File_path_for_TotalSegmentator'].to_list()
    b_files = df_t2[df_t2['Kernel'] == 'B']['File_path_for_TotalSegmentator'].to_list()
    c_files = df_t2[df_t2['Kernel'] == 'C']['File_path_for_TotalSegmentator'].to_list()
    d_files = df_t2[df_t2['Kernel'] == 'D']['File_path_for_TotalSegmentator'].to_list()

    btob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_2_checkpoint'],
                                            inkernel=b_files,
                                            outkernel=config_inference_b50f_t2['btob50f'])

    ctob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_2_checkpoint'],
                                            inkernel=c_files,
                                            outkernel=config_inference_b50f_t2['ctob50f'])
    
    dtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_2_checkpoint'],
                                            inkernel=d_files,
                                            outkernel=config_inference_b50f_t2['dtob50f'])

    b30ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_1_checkpoint'],
                                                inkernel=b30f_files,
                                                outkernel=config_inference_b50f_t2['b30ftob50f'])

    b80ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_1_checkpoint'],
                                                inkernel=b80f_files,
                                                outkernel=config_inference_b50f_t2['b80ftob50f'])
    
    bonetob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_1_checkpoint'],
                                                inkernel=bone_files,  
                                                outkernel=config_inference_b50f_t2['bonetob50f'])

    lungtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_1_checkpoint'],
                                                inkernel=lung_files,
                                                outkernel=config_inference_b50f_t2['lungtob50f'])
    
    stdtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inference_b50f_t2['stage_1_checkpoint'],
                                                inkernel=std_files,
                                                outkernel=config_inference_b50f_t2['stdtob50f'])
    
    b80ftob50f.generate_images("G_B80f_encoder", "G_B50f_decoder")
    btob50f.generate_images("G_B_encoder", "G_B50f_decoder")
    dtob50f.generate_images("G_D_encoder", "G_B50f_decoder")
    lungtob50f.generate_images("G_LUNG_encoder", "G_B50f_decoder")
    ctob50f.generate_images("G_C_encoder", "G_B50f_decoder")

    #b30ftob50f.generate_images("G_B30f_encoder", "G_B50f_decoder")
    #bonetob50f.generate_images("G_BONE_encoder", "G_B50f_decoder")
    #stdtob50f.generate_images("G_STD_encoder", "G_B50f_decoder")

