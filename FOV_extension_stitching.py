import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from typing import Union
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from nibabel.affines import voxel_sizes
import argparse
from joblib import Parallel, delayed
import pandas as pd

AIR_HU = -1024.0

def center_paste_if_aligned(orig_path, ext_path, out_path, default_val=AIR_HU):
    # 1) load (WE DO NOT RESAMPLE THE ORIGINAL)
    orig = nib.load(orig_path)            # e.g., 512x512, ~0.4 mm
    ext  = nib.load(ext_path)             # e.g., 256x256 @ 1 mm

    # 2) resample ONLY the extended image to the ORIGINAL VOXEL SIZES
    vs_orig = tuple(voxel_sizes(orig.affine))
    ext_rs  = resample_to_output(ext, voxel_sizes=(vs_orig[0]/2, vs_orig[1]/2, vs_orig[2]), order=3, cval=default_val)

    # 4) center-paste original into the resampled-extended canvas
    E = np.asarray(ext_rs.get_fdata(dtype=np.float32))
    # E = np.flip(E, 0)
    O = np.asarray(orig.get_fdata(dtype=np.float32))
    if E.shape[2] != O.shape[2]:
        # if z differs, you may choose to center on z as well
        # simple center on all 3 dims:
        pass

    H_orig, W_orig = O.shape[0], O.shape[1]
    Y, X = np.ogrid[:H_orig, :W_orig]
    cy, cx = H_orig // 2, W_orig // 2
    dist2 = (Y - cy)**2 + (X - cx)**2
    radius = int(min(H_orig, W_orig) * 0.5) - 3  # your margin
    mask2d = (dist2 <= radius**2)  # bool shape (H_orig, W_orig)

    # 2) Broadcast that 2D mask across z to get a 3D mask matching the ORIGINAL block
    region_shape = O.shape
    mask3d = np.broadcast_to(mask2d[..., None], region_shape)  # bool (Ox, Oy, Oz)

    # 3) Work on the sub-volume where the original will be pasted
    start = [(E.shape[i] - O.shape[i]) // 2 for i in range(3)]
    end   = [start[i] + O.shape[i] for i in range(3)]
    sx, sy, sz = start
    ex, ey, ez = end

    # safety
    if any(v < 0 for v in (sx, sy, sz)) or ex > E.shape[0] or ey > E.shape[1] or ez > E.shape[2]:
        raise ValueError("Center placement is out of bounds. Check shapes/alignment.")

    # 4) Copy the extended subregion, overwrite masked voxels with ORIGINAL, then put it back
    subE = E[sx:ex, sy:ey, sz:ez].copy()            # shape (Ox, Oy, Oz)
    subO = O                                        # same shape
    subE[mask3d] = subO[mask3d]                     # masked overwrite
    E[sx:ex, sy:ey, sz:ez] = subE                   # write back

    # 5) save (keep the extended affine, which represents the larger FOV)
    E = np.flip(E, 0)
    E = np.clip(E, a_min = AIR_HU, a_max=None)
    out_img = nib.Nifti1Image(E.astype(np.float32), ext_rs.affine, ext_rs.header)

    out_img.affine[:3, 0] *= 2.0  # x column
    out_img.affine[:3, 1] *= 2.0

    new_vs = voxel_sizes(out_img.affine)
    out_img.header['pixdim'][1:4] = new_vs

    nib.save(out_img, out_path)
    print("Saved FOV extended original resoltion image to", out_path)



def bucket_dir(base: Union[str, Path], id_: int, bucket: int = 1000, prefix: str = "sub", pad: int = None) -> Path:
    """
    Returns the subfolder path for a given integer ID.
    Example: id_=1234, bucket=1000 -> sub_1000-1999
    """
    base = Path(base)
    # floor-division works for negatives too (e.g., -1 // 1000 == -1)
    start = (id_ // bucket) * bucket
    end = start + bucket - 1

    def fmt(n: int) -> str:
        return str(n).zfill(pad) if pad else str(n)

    name = f"{prefix}_{fmt(start)}-{fmt(end)}"
    return base / name

output_root = 'path/to/output/folder'
data_root = 'path/to/input/folder'
study_names = sorted(os.listdir(data_root))
for study_name in study_names:
    dataset_dir = os.path.join(data_root, study_name)
    image_paths = sorted(glob(os.path.join(dataset_dir, '*.nii.gz')))

    print(f'Found {len(image_paths)} images in {dataset_dir}')

    for img_id, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing"):
        out_path = bucket_dir(Path(output_root) / Path(f'{study_name}_fov'), img_id, bucket=1000)
        out_path.mkdir(parents=True, exist_ok=True)

        image_path_fov = out_path / os.path.basename(image_path.replace('.nii.gz', '_fov_extended.nii.gz'))

        center_paste_if_aligned(
            orig_path = image_path,
            ext_path = image_path_fov,
            out_path = out_path / os.path.basename(image_path.replace('.nii.gz', '_fov_extended_orig_res.nii.gz'))
        )
