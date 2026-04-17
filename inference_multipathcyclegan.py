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
from Emphysemamodel.lungmask import ProcessLungMask
import logging
from joblib import Parallel, delayed



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


logger = logging.getLogger()#For emphysema analysis

class GenerateInferenceMultipathGAN:
    def __init__(self, config, input_encoder, output_decoder, inkernel, outkernel, inct_dir_synthetic, lung_mask_dir, project_dir):
        self.config = config
        self.input_encoder = input_encoder #Must be a path to a checkpoint (.pth)
        self.output_decoder = output_decoder #Must be path to a checkpoint (.pth)
        self.inkernel = inkernel
        self.outkernel = outkernel
        self.inct_dir_synthetic = inct_dir_synthetic #Synthetic images directory
        self.lung_mask_dir = lung_mask_dir #For emphysema analysis
        self.project_dir = project_dir #Output emphysema directory for the synthetic images

    def generate_images(self, enc, dec):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = torch.load(self.input_encoder)[enc]
        decoder = torch.load(self.output_decoder)[dec]
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

        in_nii_path = glob(os.path.join(self.config[self.inkernel], '*.nii.gz')) #Find nifti images in the specific location.
        out_nii = self.outkernel
        print(in_nii_path, out_nii)
        os.makedirs(out_nii, exist_ok=True)
        print(f'Identify {len(in_nii_path)} scans (.nii.gz)')

        #Set eval mode on and make sure that the gradients are off.
        with torch.no_grad():
            resencode.eval()
            resdecode.eval()
            for nii_path in tqdm(in_nii_path, total = len(in_nii_path)):
                test_dataset = InferenceDataloader(nii_path) #Load the volume into the dataloader
                test_dataset.load_nii()
                test_dataloader = DataLoader(dataset=test_dataset, batch_size = 64, shuffle=False, num_workers=8) #returns the pid, normalized data and the slice index
                converted_scan_idx_slice_map = {}
                for i, data in enumerate(test_dataloader):
                    pid = data['pid']
                    norm_data = data['normalized_data'].float().to(device) #Data on the device
                    latent = resencode(norm_data)
                    fake_image = resdecode(latent) #fake image generated. this is a tensor which needs to be converted to numpy array
                    fake_image_numpy = fake_image.cpu().numpy()
                    slice_idx_list = data['slice'].data.cpu().numpy().tolist()
                    for idx, slice_index in enumerate(slice_idx_list):
                        converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :] #Dictionary with all the predictions

                nii_file_name = os.path.basename(nii_path)
                converted_image = os.path.join(out_nii, nii_file_name)
                test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
                print(f"{nii_file_name} converted!")

    def emphysema_analysis(self):
        emph_analyze = EmphysemaAnalysis(in_ct_dir=self.inct_dir_synthetic, 
                                         lung_mask_dir=self.lung_mask_dir,
                                         project_dir=self.project_dir)
        # emph_analyze.generate_lung_mask()
        emph_analyze.get_emphysema_mask()
        emph_analyze.get_emphysema_measurement()

    def generate_single_image(self, enc, dec):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = torch.load(self.input_encoder)[enc]
        decoder = torch.load(self.output_decoder)[dec]
        encoderdict = OrderedDict()
        decoderdict = OrderedDict()
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

        #Set eval mode on and make sure that the gradients are off.
        with torch.no_grad():
            resencode.eval()
            resdecode.eval()
            test_dataset = InferenceDataloader(in_nii_path) #Load the volume into the dataloader
            test_dataset.load_nii()
            test_dataloader = DataLoader(dataset=test_dataset, batch_size = 25, shuffle=False, num_workers=4) #returns the pid, normalized data and the slice index
            converted_scan_idx_slice_map = {}
            for i, data in enumerate(test_dataloader):
                pid = data['pid']
                norm_data = data['normalized_data'].float().to(device) #Data on the device
                latent = resencode(norm_data)
                fake_image = resdecode(latent) #fake image generated. this is a tensor which needs to be converted to numpy array
                fake_image_numpy = fake_image.cpu().numpy()
                slice_idx_list = data['slice'].data.cpu().numpy().tolist()
                for idx, slice_index in enumerate(slice_idx_list):
                    converted_scan_idx_slice_map[slice_index] = fake_image_numpy[idx, 0, :, :] #Dictionary with all the predictions

            nii_file_name = os.path.basename(in_nii_path)
            converted_image = os.path.join(out_nii, nii_file_name)
            test_dataset.save_scan(converted_scan_idx_slice_map, converted_image)
            print(f"{nii_file_name} converted!")


class EmphysemaAnalysis:
    def __init__(self, in_ct_dir, lung_mask_dir, project_dir):
        self.in_ct_dir = in_ct_dir #Input directory with synthetic images
        self.lung_mask_dir = lung_mask_dir # Lung mask dir on original validation data
        self.project_dir = project_dir #Output dir for emphysema

    def _generate_lung_mask_config(self):
        return {
            'input': {
                'ct_dir': self.in_ct_dir
            },
            'output': {
                'root_dir': self.project_dir,
                'if_overwrite': True
            },
            'model': {
                #'model_lung_mask': 
                'model_lung_mask': 'EmphysemaModelCheckpoints/lung_mask'
            }
        }

    def generate_lung_mask(self):
        """
        Preprocessing, generating masks, level prediction, get the TCI evaluation, etc.
        :return:
        """
        # logger.info(f'##### Start preprocess #####')
        config_preprocess = self._generate_lung_mask_config()
        logger.info(f'Get lung mask\n')
        lung_mask_generator = ProcessLungMask(config_preprocess)
        lung_mask_generator.run()

    def get_emphysema_mask(self):
        print(f'Generate emphysema masks')
        lung_mask_dir = os.path.join(self.lung_mask_dir, 'lung_mask')

        emph_threshold = -950
        ct_list = os.listdir(self.in_ct_dir)

        emph_mask_dir = os.path.join(self.project_dir, 'emphysema')
        os.makedirs(emph_mask_dir, exist_ok=True)

        def _process_single_case(ct_file_name):
            in_ct = os.path.join(self.in_ct_dir, ct_file_name)
            lung_mask = os.path.join(lung_mask_dir, ct_file_name)

            ct_img = nib.load(in_ct)
            lung_img = nib.load(lung_mask)

            ct_data = ct_img.get_fdata()
            lung_data = lung_img.get_fdata()

            emph_data = np.zeros(ct_data.shape, dtype=int)
            emph_data[(ct_data < emph_threshold) & (lung_data > 0)] = 1

            emph_img = nib.Nifti1Image(emph_data,
                                       affine=ct_img.affine,
                                       header=ct_img.header)
            emph_path = os.path.join(emph_mask_dir, ct_file_name)
            nib.save(emph_img, emph_path)

        Parallel(
            n_jobs=10,
            prefer='threads'
        )(delayed(_process_single_case)(ct_file_name)
          for ct_file_name in tqdm(ct_list, total=len(ct_list)))

    def get_emphysema_measurement(self):
        lung_mask_dir = os.path.join(self.lung_mask_dir, 'lung_mask')
        emph_mask_dir = os.path.join(self.project_dir, 'emphysema')

        ct_file_list = os.listdir(lung_mask_dir)
        record_list = []
        for ct_file_name in ct_file_list:
            pid = ct_file_name.replace('.nii.gz', '')

            lung_mask = nib.load(os.path.join(lung_mask_dir, ct_file_name)).get_fdata()
            emph_mask = nib.load(os.path.join(emph_mask_dir, ct_file_name)).get_fdata()

            emph_score = 100. * np.count_nonzero(emph_mask) / np.count_nonzero(lung_mask)

            record_list.append({
                'pid': pid,
                'emph_score': emph_score
            })

        emph_score_df = pd.DataFrame(record_list)
        emph_score_csv = os.path.join(self.project_dir, 'emph.csv')
        print(f'Save to {emph_score_csv}')
        emph_score_df.to_csv(emph_score_csv, index=False)


config_validation = {
    "siemens_hard":"/NLST_harmonization_validation_data/B30f_B50f/B50f_masked",
    "siemens_soft": "/NLST_harmonization_validation_data/B30f_B50f/B30f_masked",
    "ge_bone": "/NLST_harmonization_validation_data/STANDARD_BONE/BONE",
    "ge_standard_bone": "/NLST_harmonization_validation_data/STANDARD_BONE/STANDARD",
    "ge_lung": "/NLST_harmonization_validation_data/STANDARD_LUNG/LUNG",
    "ge_standard_lung": "/NLST_harmonization_validation_data/STANDARD_LUNG/STANDARD",
    "siemens_b80f": "/NLST_harmonization_validation_data/B80f/B80f_masked",
    "siemens_b30f_b80f": "/NLST_harmonization_validation_data/B30f_B80f/B30f_masked",
    "b50ftob30f":"/validation_multipath_NLST/validation/B50ftoB30f",
    "b80ftob30f": "/validation_multipath_NLST/validation/B80ftoB30f",
    "bonetostandard": "/validation_multipath_NLST/validation/BONEtoSTANDARD",
    "lungtostandard": "/validation_multipath_NLST/validation/LUNGtoSTANDARD",
    "bonetob30f": "/validation_multipath_NLST/validation/BONEtoB30f",
    "lungtob30f": "/validation_multipath_NLST/validation/LUNGtoB30f",
    "stdtob30f": "/validation_multipath_NLST/validation/STDtoB30f",
    "checkpoint_till_epoch_72": "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_One",
    "remaining_checkpoints": "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_One_continue_train"
}


def validation_synthesis():
    for i in tqdm(range(2, 74, 2)):
        print(f"Synthesizing images for epoch {i}")
        b50f_encoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        b80f_encoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        b30f_encoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        bone_encoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        lung_encoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        std_encoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")

        b50f_decoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        b80f_decoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        b30f_decoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        bone_decoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        lung_decoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")
        std_decoder = os.path.join(config_validation["checkpoint_till_epoch_72"], str(i) + "_net_gendisc_weights.pth")

        b50ftob30f = os.path.join(config_validation["b50ftob30f"], "epoch_" + str(i))
        b80ftob30f = os.path.join(config_validation["b80ftob30f"], "epoch_" + str(i))
        bonetostandard = os.path.join(config_validation["bonetostandard"], "epoch_" + str(i))
        lungtostandard = os.path.join(config_validation["lungtostandard"], "epoch_" + str(i))
        bonetob30f = os.path.join(config_validation["bonetob30f"], "epoch_" + str(i))
        lungtob30f = os.path.join(config_validation["lungtob30f"], "epoch_" + str(i))
        stdtob30f = os.path.join(config_validation["stdtob30f"], "epoch_" + str(i))

        os.makedirs(b50ftob30f, exist_ok=True)
        os.makedirs(b80ftob30f, exist_ok=True)
        os.makedirs(bonetostandard, exist_ok=True)
        os.makedirs(lungtostandard, exist_ok=True)
        os.makedirs(bonetob30f, exist_ok=True)
        os.makedirs(lungtob30f, exist_ok=True)
        os.makedirs(stdtob30f, exist_ok=True)

        validate_b50ftob30f = GenerateInferenceMultipathGAN(config_validation, b50f_encoder, b30f_decoder, "siemens_hard", b50ftob30f, b50ftob30f)
        validate_b80ftob30f = GenerateInferenceMultipathGAN(config_validation, b80f_encoder, b30f_decoder, "siemens_b80f", b80ftob30f, b80ftob30f)
        validate_bonetostandard = GenerateInferenceMultipathGAN(config_validation, bone_encoder, std_decoder, "ge_bone", bonetostandard, bonetostandard)
        validate_lungtostandard = GenerateInferenceMultipathGAN(config_validation, lung_encoder, std_decoder, "ge_lung", lungtostandard, lungtostandard)

        validate_bonetob30f = GenerateInferenceMultipathGAN(config_validation, bone_encoder, b30f_decoder, "ge_bone", bonetob30f, bonetob30f)
        validate_lungtob30f = GenerateInferenceMultipathGAN(config_validation, lung_encoder, b30f_decoder,"ge_lung", lungtob30f, lungtob30f)
        validate_stdtob30f = GenerateInferenceMultipathGAN(config_validation, std_encoder, b30f_decoder, "ge_standard_bone", stdtob30f, stdtob30f)

        validate_b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder")
        validate_b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder")
        validate_bonetostandard.generate_images("G_BONE_encoder", "G_STD_decoder")
        validate_lungtostandard.generate_images("G_LUNG_encoder", "G_STD_decoder")
        validate_bonetob30f.generate_images("G_BONE_encoder", "G_B30f_decoder")
        validate_lungtob30f.generate_images("G_LUNG_encoder", "G_B30f_decoder")
        validate_stdtob30f.generate_images("G_STD_encoder", "G_B30f_decoder")


def validation_synthesis_continued():
    for i in tqdm(range(74, 202, 2)):
        print(f"Synthesizing images for epoch {i}")
        b50f_encoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b80f_encoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b30f_encoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        bone_encoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        lung_encoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        std_encoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")

        b50f_decoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b80f_decoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b30f_decoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        bone_decoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        lung_decoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        std_decoder = os.path.join(config_validation["remaining_checkpoints"], str(i) + "_net_gendisc_weights.pth")

        b50ftob30f = os.path.join(config_validation["b50ftob30f"], "epoch_" + str(i))
        b80ftob30f = os.path.join(config_validation["b80ftob30f"], "epoch_" + str(i))
        bonetostandard = os.path.join(config_validation["bonetostandard"], "epoch_" + str(i))
        lungtostandard = os.path.join(config_validation["lungtostandard"], "epoch_" + str(i))
        bonetob30f = os.path.join(config_validation["bonetob30f"], "epoch_" + str(i))
        lungtob30f = os.path.join(config_validation["lungtob30f"], "epoch_" + str(i))
        stdtob30f = os.path.join(config_validation["stdtob30f"], "epoch_" + str(i))

        os.makedirs(b50ftob30f, exist_ok=True)
        os.makedirs(b80ftob30f, exist_ok=True)
        os.makedirs(bonetostandard, exist_ok=True)
        os.makedirs(lungtostandard, exist_ok=True)
        os.makedirs(bonetob30f, exist_ok=True)
        os.makedirs(lungtob30f, exist_ok=True)
        os.makedirs(stdtob30f, exist_ok=True)

        validate_b50ftob30f = GenerateInferenceMultipathGAN(config_validation, b50f_encoder, b30f_decoder, "siemens_hard", b50ftob30f, b50ftob30f)
        validate_b80ftob30f = GenerateInferenceMultipathGAN(config_validation, b80f_encoder, b30f_decoder, "siemens_b80f", b80ftob30f, b80ftob30f)
        validate_bonetostandard = GenerateInferenceMultipathGAN(config_validation, bone_encoder, std_decoder, "ge_bone", bonetostandard, bonetostandard)
        validate_lungtostandard = GenerateInferenceMultipathGAN(config_validation, lung_encoder, std_decoder, "ge_lung", lungtostandard, lungtostandard)

        validate_bonetob30f = GenerateInferenceMultipathGAN(config_validation, bone_encoder, b30f_decoder, "ge_bone", bonetob30f, bonetob30f)
        validate_lungtob30f = GenerateInferenceMultipathGAN(config_validation, lung_encoder, b30f_decoder, "ge_lung", lungtob30f, lungtob30f)
        validate_stdtob30f = GenerateInferenceMultipathGAN(config_validation, std_encoder, b30f_decoder, "ge_standard_bone", stdtob30f, stdtob30f)

        validate_b50ftob30f.generate_images("G_B50f_encoder", "G_B30f_decoder")
        validate_b80ftob30f.generate_images("G_B80f_encoder", "G_B30f_decoder")
        validate_bonetostandard.generate_images("G_BONE_encoder", "G_STD_decoder")
        validate_lungtostandard.generate_images("G_LUNG_encoder", "G_STD_decoder")
        validate_bonetob30f.generate_images("G_BONE_encoder", "G_B30f_decoder")
        validate_lungtob30f.generate_images("G_LUNG_encoder", "G_B30f_decoder")
        validate_stdtob30f.generate_images("G_STD_encoder", "G_B30f_decoder")



def validation_synthesis_stage_2():
    for i in tqdm(range(2, 202, 2)):
        print(f"Synthesizing images for epoch {i}")
        b_decoder = os.path.join(config_validation_stage_2["checkpoints"], str(i) + "_net_gendisc_weights.pth")
        c_decoder = os.path.join(config_validation_stage_2["checkpoints"], str(i) + "_net_gendisc_weights.pth")
        d_encoder = os.path.join(config_validation_stage_2["checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b_encoder = os.path.join(config_validation_stage_2["checkpoints"], str(i) + "_net_gendisc_weights.pth")
        c_encoder = os.path.join(config_validation_stage_2["checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b30f_decoder = os.path.join(config_validation_stage_2["checkpoints"], str(i) + "_net_gendisc_weights.pth")

        b_to_b30f = os.path.join(config_validation_stage_2["btob30f"], "epoch_" + str(i)) #B encoder, B30f decoder
        c_to_b30f = os.path.join(config_validation_stage_2["ctob30f"], "epoch_" + str(i)) #C encoder, B30f decoder
        d_to_b30f = os.path.join(config_validation_stage_2["dtob30f"], "epoch_" + str(i)) #D encoder, B30f decoder
        d_to_c = os.path.join(config_validation_stage_2["dtoc"], "epoch_" + str(i)) #D encoder, C decoder
        d_to_b = os.path.join(config_validation_stage_2["dtob"], "epoch_" + str(i)) #D encoder, B decoder

        b_to_b30f_emph = os.path.join(config_validation_stage_2["btob30f_emph"], "epoch_" + str(i))
        c_to_b30f_emph = os.path.join(config_validation_stage_2["ctob30f_emph"], "epoch_" + str(i))
        d_to_b30f_emph = os.path.join(config_validation_stage_2["dtob30f_emph"], "epoch_" + str(i))
        d_to_c_emph = os.path.join(config_validation_stage_2["dtoc_emph"], "epoch_" + str(i))
        d_to_b_emph = os.path.join(config_validation_stage_2["dtob_emph"], "epoch_" + str(i))

        os.makedirs(b_to_b30f, exist_ok=True)
        os.makedirs(c_to_b30f, exist_ok=True)
        os.makedirs(d_to_b30f, exist_ok=True)
        os.makedirs(d_to_c, exist_ok=True)
        os.makedirs(d_to_b, exist_ok=True)

        btob30f_gen = GenerateInferenceMultipathGAN(config_validation_stage_2, b_encoder, b30f_decoder, "philips_b", b_to_b30f, b_to_b30f, config_validation_stage_2["philips_b_lung_mask"], b_to_b30f_emph)
        ctob30f_gen = GenerateInferenceMultipathGAN(config_validation_stage_2, c_encoder, b30f_decoder, "philips_c", c_to_b30f, c_to_b30f, config_validation_stage_2["philips_c_lung_mask"], c_to_b30f_emph)
        dtob30f_gen = GenerateInferenceMultipathGAN(config_validation_stage_2, d_encoder, b30f_decoder, "philips_c_d", d_to_b30f, d_to_b30f, config_validation_stage_2["philips_c_d_lung_mask"], d_to_b30f_emph) #Use the 100 subjects from Philips D
        dtoc_gen = GenerateInferenceMultipathGAN(config_validation_stage_2, d_encoder, c_decoder, "philips_c_d", d_to_c, d_to_c, config_validation_stage_2["philips_c_d_lung_mask"], d_to_c_emph)
        dtob_gen = GenerateInferenceMultipathGAN(config_validation_stage_2, d_encoder, b_decoder, "philips_b_d", d_to_b, d_to_b, config_validation_stage_2["philips_b_d_lung_mask"], d_to_b_emph)

        btob30f_gen.generate_images("G_B_encoder", "G_B30f_decoder")
        btob30f_gen.emphysema_analysis()

        dtob_gen.generate_images("G_D_encoder", "G_B_decoder")
        dtob_gen.emphysema_analysis()

        ctob30f_gen.generate_images("G_C_encoder", "G_B30f_decoder")
        ctob30f_gen.emphysema_analysis()

        dtob30f_gen.generate_images("G_D_encoder", "G_B30f_decoder")
        dtob30f_gen.emphysema_analysis()

        dtoc_gen.generate_images("G_D_encoder", "G_C_decoder")
        dtoc_gen.emphysema_analysis()


def validation_harmonize_to_hard_kernel():
    #Validate all kernels harmonized to a reference hard kernel (Siemens B50f in this case)
    config_hard_kernel_validation = {
        "stage1_checkpoints": "/NLST_Anatomy_constrained_multipath_cycleGAN_segloss_equal_to_1/NLST_MultipathGAN_with_anatomy_guidance_Stage_One",
        "stage2_checkpoints": "/NLST_MultipathGAN_with_anatomy_guidance_Stage_two_henrix_2000images_per_epoch",
        "siemens_b30f": "/validation_multipath_NLST/NLST_harmonization_validation_data/B30f_B50f/B30f_masked", #B30f
        "siemens_b80f": "/validation_multipath_NLST/NLST_harmonization_validation_data/B30f_B80f/B80f_masked", #B80f
        "ge_bone": "/validation_multipath_NLST/NLST_harmonization_validation_data/STANDARD_BONE/BONE", #BONE
        "ge_standard": "/validation_multipath_NLST/NLST_harmonization_validation_data/STANDARD_BONE/STANDARD", #STANDARD
        "ge_lung": "/validation_multipath_NLST/NLST_harmonization_validation_data/STANDARD_LUNG/LUNG", #LUNG
        "philips_b": "/validation_multipath_NLST/NLST_harmonization_validation_data/B_D/B", #B
        "philips_c": "/validation_multipath_NLST/NLST_harmonization_validation_data/C_D/C", #C
        "philips_d": "/validation_multipath_NLST/NLST_harmonization_validation_data/C_D/D", #D
        "b30ftob50f": "/validation_multipath_NLST/validation_stage1/B30ftoB50f", 
        "b30ftob50f_emph": "/validation_multipath_NLST/validation_stage1/B30ftoB50f_emphysema",
        "b80ftob50f": "/validation_multipath_NLST/validation_stage1/B80ftoB50f", 
        "b80ftob50f_emph": "/validation_multipath_NLST/validation_stage1/B80ftoB50f_emphysema",
        "bonetob50f": "/validation_multipath_NLST/validation_stage1/BONEtoB50f",
        "bonetob50f_emph": "/validation_multipath_NLST/validation_stage1/BONEtoB50f_emphysema",
        "lungtob50f": "/validation_multipath_NLST/validation_stage1/LUNGtoB50f",
        "lungtob50f_emph": "/validation_multipath_NLST/validation_stage1/LUNGtoB50f_emphysema",
        "stdtob50f": "/validation_multipath_NLST/validation_stage1/STANDARDtoB50f",
        "stdtob50f_emph": "/validation_multipath_NLST/validation_stage1/STANDARDtoB50f_emphysema",
        "btob50f": "/validation_multipath_NLST/validation_stage2/BtoB50f",
        "btob50f_emph" : "/validation_multipath_NLST/validation_stage2/BtoB50f_emphysema",
        "ctob50f": "/validation_multipath_NLST/validation_stage2/CtoB50f",
        "ctob50f_emph": "/validation_multipath_NLST/validation_stage2/CtoB50f_emphysema",
        "dtob50f": "/validation_multipath_NLST/validation_stage2/DtoB50f",
        "dtob50f_emph": "/validation_multipath_NLST/validation_stage2/DtoB50f_emphysema",
        "b30f_lung_mask": "/NLST_harmonization_validation_data/B30f_B50f/B30f_masked_emphysema",
        "b80f_lung_mask": "/NLST_harmonization_validation_data/B30f_B80f/B80f_masked_emphysema",
        "bone_lung_mask": "/NLST_harmonization_validation_data/STANDARD_BONE/BONE_emphysema",
        "std_lung_mask": "/NLST_harmonization_validation_data/STANDARD_BONE/STANDARD_emphysema",
        "lung_lung_mask": "/NLST_harmonization_validation_data/STANDARD_LUNG/LUNG_emphysema",
        "b_lung_mask": "/NLST_harmonization_validation_data/B_D/B_emphysema",
        "c_lung_mask": "/NLST_harmonization_validation_data/C_D/C_emphysema",
        "d_lung_mask": "/NLST_harmonization_validation_data/C_D/D_emphysema"
    }
    
    for i in tqdm(range(2, 202, 2)):
        b30f_encoder = os.path.join(config_hard_kernel_validation["stage1_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b80f_encoder = os.path.join(config_hard_kernel_validation["stage1_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        bone_encoder = os.path.join(config_hard_kernel_validation["stage1_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        lung_encoder = os.path.join(config_hard_kernel_validation["stage1_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        std_encoder = os.path.join(config_hard_kernel_validation["stage1_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b_encoder = os.path.join(config_hard_kernel_validation["stage2_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        c_encoder = os.path.join(config_hard_kernel_validation["stage2_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        d_encoder = os.path.join(config_hard_kernel_validation["stage2_checkpoints"], str(i) + "_net_gendisc_weights.pth")
        b50f_decoder = os.path.join(config_hard_kernel_validation["stage1_checkpoints"], str(i) + "_net_gendisc_weights.pth")

        b30ftob50f = os.path.join(config_hard_kernel_validation["b30ftob50f"], "epoch_" + str(i))
        b80ftob50f = os.path.join(config_hard_kernel_validation["b80ftob50f"], "epoch_" + str(i))
        bonetob50f = os.path.join(config_hard_kernel_validation["bonetob50f"], "epoch_" + str(i))
        lungtob50f = os.path.join(config_hard_kernel_validation["lungtob50f"], "epoch_" + str(i))
        stdtob50f = os.path.join(config_hard_kernel_validation["stdtob50f"], "epoch_" + str(i))
        btob50f = os.path.join(config_hard_kernel_validation["btob50f"], "epoch_" + str(i))
        ctob50f = os.path.join(config_hard_kernel_validation["ctob50f"], "epoch_" + str(i))
        dtob50f = os.path.join(config_hard_kernel_validation["dtob50f"], "epoch_" + str(i))

        b30ftob50f_emph = os.path.join(config_hard_kernel_validation["b30ftob50f_emph"], "epoch_" + str(i))
        b80ftob50f_emph = os.path.join(config_hard_kernel_validation["b80ftob50f_emph"], "epoch_" + str(i))
        bonetob50f_emph = os.path.join(config_hard_kernel_validation["bonetob50f_emph"], "epoch_" + str(i))
        lungtob50f_emph = os.path.join(config_hard_kernel_validation["lungtob50f_emph"], "epoch_" + str(i))
        stdtob50f_emph = os.path.join(config_hard_kernel_validation["stdtob50f_emph"], "epoch_" + str(i))
        btob50f_emph = os.path.join(config_hard_kernel_validation["btob50f_emph"], "epoch_" + str(i))
        ctob50f_emph = os.path.join(config_hard_kernel_validation["ctob50f_emph"], "epoch_" + str(i))
        dtob50f_emph = os.path.join(config_hard_kernel_validation["dtob50f_emph"], "epoch_" + str(i))

        b30ftob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                       b30f_encoder, b50f_decoder, "siemens_b30f", 
                                                       b30ftob50f, b30ftob50f, 
                                                       config_hard_kernel_validation["b30f_lung_mask"], 
                                                       b30ftob50f_emph)
        
        b80ftob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                       b80f_encoder, b50f_decoder, "siemens_b80f", 
                                                       b80ftob50f, b80ftob50f, 
                                                       config_hard_kernel_validation["b80f_lung_mask"], 
                                                       b80ftob50f_emph)
        
        bonetob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                        bone_encoder, b50f_decoder, "ge_bone",
                                                        bonetob50f, bonetob50f,
                                                        config_hard_kernel_validation["bone_lung_mask"],
                                                        bonetob50f_emph)
        
        stdtob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                      std_encoder, b50f_decoder, "ge_standard",
                                                      stdtob50f, stdtob50f,
                                                      config_hard_kernel_validation["std_lung_mask"],
                                                      stdtob50f_emph)
        
        lungtob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                       lung_encoder, b50f_decoder, "ge_lung",
                                                       lungtob50f, lungtob50f,
                                                       config_hard_kernel_validation["lung_lung_mask"],
                                                       lungtob50f_emph)

        btob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                     b_encoder, b50f_decoder, "philips_b",
                                                     btob50f, btob50f,
                                                     config_hard_kernel_validation["b_lung_mask"],
                                                     btob50f_emph)
        
        ctob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                    c_encoder, b50f_decoder, "philips_c",
                                                    ctob50f, ctob50f,
                                                    config_hard_kernel_validation["c_lung_mask"],
                                                    ctob50f_emph)
        
        dtob50f_gen = GenerateInferenceMultipathGAN(config_hard_kernel_validation,
                                                    d_encoder, b50f_decoder, "philips_d",
                                                    dtob50f, dtob50f,
                                                    config_hard_kernel_validation["d_lung_mask"],
                                                    dtob50f_emph)
        
        b30ftob50f_gen.generate_images("G_B30f_encoder", "G_B50f_decoder")
        b30ftob50f_gen.emphysema_analysis()

        b80ftob50f_gen.generate_images("G_B80f_encoder", "G_B50f_decoder")
        b80ftob50f_gen.emphysema_analysis()

        bonetob50f_gen.generate_images("G_BONE_encoder", "G_B50f_decoder")
        bonetob50f_gen.emphysema_analysis()

        stdtob50f_gen.generate_images("G_STD_encoder", "G_B50f_decoder")
        stdtob50f_gen.emphysema_analysis()

        lungtob50f_gen.generate_images("G_LUNG_encoder", "G_B50f_decoder")
        lungtob50f_gen.emphysema_analysis()

        btob50f_gen.generate_images("G_B_encoder", "G_B50f_decoder")
        btob50f_gen.emphysema_analysis()

        ctob50f_gen.generate_images("G_C_encoder", "G_B50f_decoder")
        ctob50f_gen.emphysema_analysis()

        dtob50f_gen.generate_images("G_D_encoder", "G_B50f_decoder")
        dtob50f_gen.emphysema_analysis()

validation_harmonize_to_hard_kernel()