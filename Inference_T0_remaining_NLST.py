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


def harmonize_NLST_T0_to_B30f():
    config_inference_stage1 = {
        'stage1_checkpoint': "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/152_net_gendisc_weights.pth",
        'stage2_checkpoint_temp': "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/56_net_gendisc_weights.pth", #Temp epoch, have not run validation on the other model which needs to be run
        'b50ftob30f': "/NLST_harmonized_images/harmonized_to_B30f/B50ftoB30f_remaining_T0",
        'b80ftob30f': "/NLST_harmonized_images/harmonized_to_B30f/B80ftoB30f_remaining_T0",
        'bonetob30f': "/NLST_harmonized_images/harmonized_to_B30f/BONEttoB30f_remaining_T0",
        'lungtob30f': "/NLST_harmonized_images/harmonized_to_B30f/LUNGtoB30f_remaining_T0",
        'btob30f': '/NLST_harmonized_images/harmonized_to_B30f/BtoB30f_temp_remaining_T0', 
        'ctob30f': '/NLST_harmonized_images/harmonized_to_B30f/CtoB30f_temp_remaining_T0',
        'dtob30f': '/NLST_harmonized_images/harmonized_to_B30f/DtoB30f_temp_remaining_T0',
        'stdtob30f': '/NLST_harmonized_images/harmonized_to_B30f/STANDARDtoB30f_remaining_T0'
    }
    df = pd.read_csv("spreadsheet_with_resampled_paths_post_QA/NLST_T0_all_38060_symlink_fixed_resampled_fixed_with_harmonization_needed_final_12_11_25.csv")

    df_only_harmonize = df[df["Harmonization_needed_to_B30f"] == "Yes"]

    b50f_files = df_only_harmonize[df_only_harmonize['Kernel'] == "B50f"]['File_path_for_TotalSegmentator'].to_list()
    b80f_files = df_only_harmonize[df_only_harmonize['Kernel'] == "B80f"]['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_only_harmonize[df_only_harmonize['Kernel'] == "BONE"]['File_path_for_TotalSegmentator'].to_list()
    lung_files = df_only_harmonize[df_only_harmonize['Kernel'] == "LUNG"]['File_path_for_TotalSegmentator'].to_list()
    standard_files = df_only_harmonize[df_only_harmonize['Kernel'] == "STANDARD"]['File_path_for_TotalSegmentator'].to_list()
    b_files = df_only_harmonize[df_only_harmonize['Kernel'] == "B"]['File_path_for_TotalSegmentator'].to_list()
    c_files = df_only_harmonize[df_only_harmonize['Kernel'] == "C"]['File_path_for_TotalSegmentator'].to_list()
    d_files = df_only_harmonize[df_only_harmonize['Kernel'] == "D"]['File_path_for_TotalSegmentator'].to_list()
    


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
                                                inkernel=standard_files,
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
    
        
    b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder") 
    b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder") 
    bonetob30f.generate_images("G_BONE_encoder", "G_B30f_decoder") 
    lungtob30f.generate_images("G_LUNG_encoder", "G_B30f_decoder") 
    btob30f.generate_images("G_B_encoder", "G_B30f_decoder")
    ctob30f.generate_images("G_C_encoder", "G_B30f_decoder")
    dtob30f.generate_images("G_D_encoder", "G_B30f_decoder")
    stdtob30f.generate_images("G_STD_encoder", "G_B30f_decoder")

# harmonize_NLST_T0_to_B30f()

def harmonize_NLST_T0_to_B50f():
    config_inferece_b50f = {
        'stage_1_checkpoint': "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train/198_net_gendisc_weights.pth",
        'stage_2_checkpoint': "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch/190_net_gendisc_weights.pth", 
        'b30ftob50f': "/NLST_harmonized_images/harmonized_to_B50f/B30ftoB50f_remaining_T0",  
        'b80ftob50f': "/NLST_harmonized_images/harmonized_to_B50f/B80ftoB50f_remaining_T0", 
        'bonetob50f': "/NLST_harmonized_images/harmonized_to_B50f/BONEtoB50f_remaining_T0", 
        'lungtob50f': "/NLST_harmonized_images/harmonized_to_B50f/LUNGtoB50f_remaining_T0", 
        'stdtob50f': "//NLST_harmonized_images/harmonized_to_B50f/STANDARDtoB50f_remaining_T0",
        'btob50f': "/NLST_harmonized_images/harmonized_to_B50f/BtoB50f_remaining_T0",
        'ctob50f': "/NLST_harmonized_images/harmonized_to_B50f/CtoB50f_remaining_T0",
        'dtob50f': "/NLST_harmonized_images/harmonized_to_B50f/DtoB50f_remaining_T0"
    }

    df = pd.read_csv("/spreadsheet_with_resampled_paths_post_QA/NLST_T0_all_38060_symlink_fixed_resampled_fixed_with_harmonization_needed_final_12_11_25.csv")

    df_only_harmonize = df[df["Harmonization_needed_to_B50f"] == "Yes"]

    b30f_files = df_only_harmonize[df_only_harmonize['Kernel'] == "B30f"]['File_path_for_TotalSegmentator'].to_list()
    b80f_files = df_only_harmonize[df_only_harmonize['Kernel'] == "B80f"]['File_path_for_TotalSegmentator'].to_list()
    bone_files = df_only_harmonize[df_only_harmonize['Kernel'] == "BONE"]['File_path_for_TotalSegmentator'].to_list()
    lung_files = df_only_harmonize[df_only_harmonize['Kernel'] == "LUNG"]['File_path_for_TotalSegmentator'].to_list()
    standard_files = df_only_harmonize[df_only_harmonize['Kernel'] == "STANDARD"]['File_path_for_TotalSegmentator'].to_list()
    b_files = df_only_harmonize[df_only_harmonize['Kernel'] == "B"]['File_path_for_TotalSegmentator'].to_list()
    c_files = df_only_harmonize[df_only_harmonize['Kernel'] == "C"]['File_path_for_TotalSegmentator'].to_list()
    d_files = df_only_harmonize[df_only_harmonize['Kernel'] == "D"]['File_path_for_TotalSegmentator'].to_list() 
    

    btob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=b_files,
                                            outkernel=config_inferece_b50f['btob50f'])
    
    ctob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=c_files,
                                            outkernel=config_inferece_b50f['ctob50f'])

    dtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_2_checkpoint'],
                                            inkernel=d_files,
                                            outkernel=config_inferece_b50f['dtob50f'])
    
    b30ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=b30f_files,
                                               outkernel=config_inferece_b50f['b30ftob50f'])
    
    b80ftob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=b80f_files,
                                               outkernel=config_inferece_b50f['b80ftob50f'])

    bonetob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=bone_files,
                                               outkernel=config_inferece_b50f['bonetob50f'])

    stdtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                               inkernel=standard_files,
                                               outkernel=config_inferece_b50f['stdtob50f'])
    
    lungtob50f = GenerateInferenceMultipathGAN(model_checkpoint=config_inferece_b50f['stage_1_checkpoint'],
                                                  inkernel=lung_files,
                                                    outkernel=config_inferece_b50f['lungtob50f'])
    
    btob50f.generate_images("G_B_encoder", "G_B50f_decoder")
    ctob50f.generate_images("G_C_encoder", "G_B50f_decoder")
    dtob50f.generate_images("G_D_encoder", "G_B50f_decoder")
    b30ftob50f.generate_images("G_B30f_encoder", "G_B50f_decoder")
    b80ftob50f.generate_images("G_B80f_encoder", "G_B50f_decoder")
    bonetob50f.generate_images("G_BONE_encoder", "G_B50f_decoder")
    lungtob50f.generate_images("G_LUNG_encoder", "G_B50f_decoder")
    stdtob50f.generate_images("G_STD_encoder", "G_B50f_decoder")

harmonize_NLST_T0_to_B50f()