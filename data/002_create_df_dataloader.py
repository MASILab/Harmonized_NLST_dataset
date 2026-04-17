import os 
import pandas as pd 
from tqdm import tqdm
import time

def create_df_training():
    main_path = "lung_data/training_data"
    save_path = "lung_data/training_data/train_dataframes"

    b50f_images = os.path.join(main_path, 'B50f_image_slices')
    b50f_masks = os.path.join(main_path, 'B50f_mask_slices')
    b30f_images = os.path.join(main_path, 'B30f_image_slices')
    b30f_masks = os.path.join(main_path, 'B30f_mask_slices')
    bone_images = os.path.join(main_path, 'BONE_image_slices')
    bone_masks = os.path.join(main_path, 'BONE_mask_slices')
    standard_images = os.path.join(main_path, 'STANDARD_image_slices')
    standard_masks = os.path.join(main_path, 'STANDARD_mask_slices')
    lung_images = os.path.join(main_path, 'LUNG_image_slices')
    lung_masks = os.path.join(main_path, 'LUNG_mask_slices')
    b80f_images = os.path.join(main_path, 'B80f_image_slices')
    b80f_masks = os.path.join(main_path, 'B80f_mask_slices')
    c_images = os.path.join(main_path, 'C_image_slices')
    c_masks = os.path.join(main_path, 'C_mask_slices')
    d_images = os.path.join(main_path, 'D_image_slices')
    d_masks = os.path.join(main_path, 'D_mask_slices')
    b_images = os.path.join(main_path, 'B_image_slices')
    b_masks = os.path.join(main_path, 'B_mask_slices')

    #Whole path to the images and masks 
    b50f_image_paths = sorted([os.path.join(b50f_images, f) for f in os.listdir(b50f_images)])
    b50f_mask_paths = sorted([os.path.join(b50f_masks, f) for f in os.listdir(b50f_masks)])
    b30f_image_paths = sorted([os.path.join(b30f_images, f) for f in os.listdir(b30f_images)])
    b30f_mask_paths = sorted([os.path.join(b30f_masks, f) for f in os.listdir(b30f_masks)])
    bone_image_paths = sorted([os.path.join(bone_images, f) for f in os.listdir(bone_images)])
    bone_mask_paths = sorted([os.path.join(bone_masks, f) for f in os.listdir(bone_masks)])
    standard_image_paths = sorted([os.path.join(standard_images, f) for f in os.listdir(standard_images)])
    standard_mask_paths = sorted([os.path.join(standard_masks, f) for f in os.listdir(standard_masks)])
    lung_image_paths = sorted([os.path.join(lung_images, f) for f in os.listdir(lung_images)])
    lung_mask_paths = sorted([os.path.join(lung_masks, f) for f in os.listdir(lung_masks)])
    b80f_image_paths = sorted([os.path.join(b80f_images, f) for f in os.listdir(b80f_images)])
    b80f_mask_paths = sorted([os.path.join(b80f_masks, f) for f in os.listdir(b80f_masks)])
    c_image_paths = sorted([os.path.join(c_images, f) for f in os.listdir(c_images)])
    c_mask_paths = sorted([os.path.join(c_masks, f) for f in os.listdir(c_masks)])
    d_image_paths = sorted([os.path.join(d_images, f) for f in os.listdir(d_images)])
    d_mask_paths = sorted([os.path.join(d_masks, f) for f in os.listdir(d_masks)])
    b_image_paths = sorted([os.path.join(b_images, f) for f in os.listdir(b_images)])
    b_mask_paths = sorted([os.path.join(b_masks, f) for f in os.listdir(b_masks)])

    #Create a dataframe for each domain
    df_b50f = pd.DataFrame({'image': b50f_image_paths, 'mask': b50f_mask_paths})
    df_b30f = pd.DataFrame({'image': b30f_image_paths, 'mask': b30f_mask_paths})
    df_bone = pd.DataFrame({'image': bone_image_paths, 'mask': bone_mask_paths})
    df_standard = pd.DataFrame({'image': standard_image_paths, 'mask': standard_mask_paths})
    df_lung = pd.DataFrame({'image': lung_image_paths, 'mask': lung_mask_paths})
    df_b80f = pd.DataFrame({'image': b80f_image_paths, 'mask': b80f_mask_paths})
    df_c = pd.DataFrame({'image': c_image_paths, 'mask': c_mask_paths})
    df_d = pd.DataFrame({'image': d_image_paths, 'mask': d_mask_paths})
    df_b = pd.DataFrame({'image': b_image_paths, 'mask': b_mask_paths})

    #Add a column for the domain name
    df_b50f['domain'] = 'B50f'
    df_b30f['domain'] = 'B30f'
    df_bone['domain'] = 'BONE'  
    df_standard['domain'] = 'STANDARD'
    df_lung['domain'] = 'LUNG'
    df_b80f['domain'] = 'B80f'
    df_c['domain'] = 'C'
    df_d['domain'] = 'D'
    df_b['domain'] = 'B'   

    #Save each dataframe to a csv file
    df_b50f.to_csv(os.path.join(save_path, 'df_b50f.csv'), index=False)
    df_b30f.to_csv(os.path.join(save_path, 'df_b30f.csv'), index=False)
    df_bone.to_csv(os.path.join(save_path, 'df_bone.csv'), index=False)
    df_standard.to_csv(os.path.join(save_path, 'df_standard.csv'), index=False)
    df_lung.to_csv(os.path.join(save_path, 'df_lung.csv'), index=False)
    df_b80f.to_csv(os.path.join(save_path, 'df_b80f.csv'), index=False)
    df_c.to_csv(os.path.join(save_path, 'df_c.csv'), index=False)
    df_d.to_csv(os.path.join(save_path, 'df_d.csv'), index=False)
    df_b.to_csv(os.path.join(save_path, 'df_b.csv'), index=False)
    print("Dataframes created and saved successfully.")

# create_df_training()

def read_dataframes_and_check_time():
    start_time = time.time()
    save_path = "lung_data/training_data/train_dataframes"
    df_b50f = pd.read_csv(os.path.join(save_path, 'df_b50f.csv'))
    df_b30f = pd.read_csv(os.path.join(save_path, 'df_b30f.csv'))
    df_bone = pd.read_csv(os.path.join(save_path, 'df_bone.csv'))
    df_standard = pd.read_csv(os.path.join(save_path, 'df_standard.csv'))
    df_lung = pd.read_csv(os.path.join(save_path, 'df_lung.csv'))
    df_b80f = pd.read_csv(os.path.join(save_path, 'df_b80f.csv'))
    df_c = pd.read_csv(os.path.join(save_path, 'df_c.csv'))
    df_d = pd.read_csv(os.path.join(save_path, 'df_d.csv'))
    df_b = pd.read_csv(os.path.join(save_path, 'df_b.csv'))

    print("Dataframes read successfully.")
    
    end_time = time.time()
    print(f"Time taken to read dataframes and process: {end_time - start_time} seconds")

read_dataframes_and_check_time()