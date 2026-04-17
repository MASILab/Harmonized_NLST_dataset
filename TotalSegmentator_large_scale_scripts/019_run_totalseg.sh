#!/bin/bash

img_path="$1"

my_img_name=$(basename "$img_path" | sed 's/\.nii\.gz$//; s/\.nii$//')


#Create output directory with the image name and save the segmentation there

output_dir="/path/to/output/dir/$my_img_name"

echo "Processing image: $my_img_name"
echo "Output directory: $output_dir"

mkdir -p "$output_dir"

# Run TotalSeg with the specified parameters
TotalSegmentator -i "$img_path" --task total -o "$output_dir/${my_img_name}_total.nii.gz" --remove_small_blobs --ml

#Run tissue types model
TotalSegmentator -i "$img_path" --task tissue_types -o "$output_dir/${my_img_name}_tissue_types.nii.gz" --remove_small_blobs --ml
