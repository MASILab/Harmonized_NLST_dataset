import pandas as pd
from Emphysemamodel.lungmask import ProcessLungMask
import logging
import os
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


logger = logging.getLogger()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
                'model_lung_mask': 'EmphysemaModelCheckpoints/lung_mask' #Path to generate the lung masks
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


config_val = {
    "siemens_b50f_lung": "/NLST_harmonization_validation_data/B30f_B50f/B50f_masked_emphysema",
    "siemens_b80f_lung": "/NLST_harmonization_validation_data/B30f_B80f/B80f_masked_emphysema",
    "ge_bone_lung": "/NLST_harmonization_validation_data/STANDARD_BONE/BONE_emphysema",
    "ge_standard_lung": "/NLST_harmonization_validation_data/STANDARD_BONE/STANDARD_emphysema",
    "ge_lung_lung": "/NLST_harmonization_validation_data/STANDARD_LUNG/LUNG_emphysema",

    "b50ftob30f": "/validation/B50ftoB30f",
    "b80ftob30f": "/validation/B80ftoB30f",
    "bonetostd": "/validation/BONEtoSTANDARD",
    "lungtostd": "/validation/LUNGtoSTANDARD",
    "bonetob30f": "/validation/BONEtoB30f",
    "lungtob30f": "/validation/LUNGtoB30f",
    "stdtob30f": "/validation/STANDARDtoB30f",

    "b50fto30f_emph": "/validation/B50ftoB30f_emphysema",
    "b80fto30f_emph": "/validation/B80ftoB30f_emphysema",
    "bonetostd_emph": "/validation/BONEtoSTANDARD_emphysema",
    "lungtostd_emph": "validation//LUNGtoSTANDARD_emphysema",
    "bonetob30f_emph": "/validation/BONEtoB30f_emphysema",
    "lungtob30f_emph": "/validation/LUNGtoB30f_emphysema",
    "stdtob30f_emph": "/validation/STANDARDtoB30f_emphysema"

}


def run_validation_emphysema():
    for i in tqdm(range(112,202,2)):
        b50ftob30f = os.path.join(config_val["b50ftob30f"], 'epoch_' + str(i))
        b80ftob30f = os.path.join(config_val["b80ftob30f"], 'epoch_' + str(i))
        bonetostd = os.path.join(config_val["bonetostd"], 'epoch_' + str(i))
        lungtostd = os.path.join(config_val["lungtostd"], 'epoch_' + str(i))
        bonetob30f = os.path.join(config_val["bonetob30f"], 'epoch_' + str(i))
        lungtob30f = os.path.join(config_val["lungtob30f"], 'epoch_' + str(i))
        stdtob30f = os.path.join(config_val["stdtob30f"], 'epoch_' + str(i))

        b50flung_mask = config_val["siemens_b50f_lung"]
        b80flung_mask = config_val["siemens_b80f_lung"]
        bonemask = config_val["ge_bone_lung"]
        lungmask = config_val["ge_standard_lung"]
        stdmask = config_val["ge_lung_lung"]

        b50ftob30f_emph = os.path.join(config_val["b50fto30f_emph"], 'epoch_' + str(i))
        b80ftob30f_emph = os.path.join(config_val["b80fto30f_emph"], 'epoch_' + str(i))
        bonetostd_emph = os.path.join(config_val["bonetostd_emph"], 'epoch_' + str(i))
        lungtostd_emph = os.path.join(config_val["lungtostd_emph"], 'epoch_' + str(i))
        bonetob30f_emph = os.path.join(config_val["bonetob30f_emph"], 'epoch_' + str(i))
        lungtob30f_emph = os.path.join(config_val["lungtob30f_emph"], 'epoch_' + str(i))
        stdtob30f_emph = os.path.join(config_val["stdtob30f_emph"], 'epoch_' + str(i))

        emph_b50ftob30f = EmphysemaAnalysis(b50ftob30f, b50flung_mask, b50ftob30f_emph)
        emph_b80ftob30f = EmphysemaAnalysis(b80ftob30f, b80flung_mask, b80ftob30f_emph)
        emph_bonetostd = EmphysemaAnalysis(bonetostd, bonemask, bonetostd_emph)
        emph_lungtostd = EmphysemaAnalysis(lungtostd, lungmask, lungtostd_emph)
        emph_bonetob30f = EmphysemaAnalysis(bonetob30f, bonemask, bonetob30f_emph)
        emph_lungtob30f = EmphysemaAnalysis(lungtob30f, lungmask, lungtob30f_emph)
        emph_stdtob30f = EmphysemaAnalysis(stdtob30f, stdmask, stdtob30f_emph)

        emph_b50ftob30f.get_emphysema_mask()
        emph_b50ftob30f.get_emphysema_measurement()

        # emph_b80ftob30f.get_emphysema_mask()
        # emph_b80ftob30f.get_emphysema_measurement()

        #emph_bonetostd.get_emphysema_mask()
        #emph_bonetostd.get_emphysema_measurement()

        # emph_lungtostd.get_emphysema_mask()
        # emph_lungtostd.get_emphysema_measurement()

        # emph_bonetob30f.get_emphysema_mask()
        # emph_bonetob30f.get_emphysema_measurement()

        # emph_lungtob30f.get_emphysema_mask()
        # emph_lungtob30f.get_emphysema_measurement()

        # emph_stdtob30f.get_emphysema_mask()
        # emph_stdtob30f.get_emphysema_measurement()

run_validation_emphysema()
