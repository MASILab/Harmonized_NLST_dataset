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
    def __init__(self, in_ct_dir, project_dir):
        self.in_ct_dir = in_ct_dir
        self.project_dir = project_dir

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
                #'model_lung_mask': '/nfs/masi/xuk9/Projects/ThoraxLevelBCA/models/lung_mask' #Original path
                'model_lung_mask': '/valiant02/masi/krishar1/EmphysemaModelCheckpoints/lung_mask'
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
        lung_mask_dir = os.path.join(self.project_dir, 'lung_mask')

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
        lung_mask_dir = os.path.join(self.project_dir, 'lung_mask')
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

# ct_dirs = [
#     "/valiant02/masi/krishar1/NLST_supplementary_grants/SPIE_2026/NLST_harmonization_validation_data/B30f_B50f/B30f_masked",
#     "/valiant02/masi/krishar1/NLST_supplementary_grants/SPIE_2026/NLST_harmonization_validation_data/B30f_B80f/B80f_masked",
#     "/valiant02/masi/krishar1/NLST_supplementary_grants/SPIE_2026/NLST_harmonization_validation_data/B30f_B80f/B30f_masked",
#     #"/valiant02/masi/krishar1/NLST_supplementary_grants/SPIE_2026/NLST_harmonization_validation_data/STANDARD_BONE/BONE",
#     #"/valiant02/masi/krishar1/NLST_supplementary_grants/SPIE_2026/NLST_harmonization_validation_data/STANDARD_LUNG/LUNG",
#     #"/valiant02/masi/krishar1/NLST_supplementary_grants/SPIE_2026/NLST_harmonization_validation_data/STANDARD_BONE/STANDARD",
#     #"/valiant02/masi/krishar1/NLST_supplementary_grants/SPIE_2026/NLST_harmonization_validation_data/STANDARD_LUNG/STANDARD",
# ]

# for ct_dir in tqdm(ct_dirs):
#     emph = EmphysemaAnalysis(in_ct_dir=ct_dir,
#                              project_dir= ct_dir + '_emphysema')

#     emph.generate_lung_mask()
#     emph.get_emphysema_mask()
#     emph.get_emphysema_measurement()

#Run emphysema analysis on the validation data using the masks from the original validation data 

# emph = EmphysemaAnalysis(in_ct_dir = '/valiant02/masi/krishar1/NLST_supplementary_grants/validation_multipath_NLST/validation/LUNGtoB30f/epoch_114',
#                          project_dir = '/valiant02/masi/krishar1/NLST_supplementary_grants/validation_multipath_NLST/validation/LUNGtoB30f_emphysema/epoch_114')

# # emph.generate_lung_mask()
# emph.get_emphysema_mask()
# emph.get_emphysema_measurement()

dirs = [ "/valiant02/masi/krishar1/NLST_supplementary_grants/validation_multipath_NLST/NLST_harmonization_validation_data/B_D/B", 
        "/valiant02/masi/krishar1/NLST_supplementary_grants/validation_multipath_NLST/NLST_harmonization_validation_data/B_D/D", 
        "/valiant02/masi/krishar1/NLST_supplementary_grants/validation_multipath_NLST/NLST_harmonization_validation_data/C_D/C",
        "/valiant02/masi/krishar1/NLST_supplementary_grants/validation_multipath_NLST/NLST_harmonization_validation_data/C_D/D"
]


for emph_dir in tqdm(dirs):
    emph = EmphysemaAnalysis(in_ct_dir = emph_dir, 
                            project_dir = emph_dir + '_emphysema')
    emph.generate_lung_mask()
    emph.get_emphysema_mask()
    emph.get_emphysema_measurement()
    # print(emph_dir + "_emphysema")