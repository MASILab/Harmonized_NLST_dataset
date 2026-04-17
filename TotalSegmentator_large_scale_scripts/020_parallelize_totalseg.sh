#!/bin/bash
#Given a full nifti path to an image, run TotalSeg and save the segmentations as a multilabel mask

 if [[ "$(hostname -f)" == "HENDRIX.vuds.vanderbilt.edu" ]]; then
        export CUDA_VISIBLE_DEVICES=1
        echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"
 fi

if [[ "$(hostname -f)" == "pinkfloyd.masi.vanderbilt.edu" ]]; then
        export CUDA_VISIBLE_DEVICES=0
        echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

fi

if [[ "$(hostname -f)" == "deeppurple.vuds.vanderbilt.edu" ]]; then
        export CUDA_VISIBLE_DEVICES=2
        echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

fi

if [[ "$(hostname -f)" == "masi-54.vuds.vanderbilt.edu" ]]; then
        export CUDA_VISIBLE_DEVICES=1
        echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"

fi

nifti_path="$1"

echo "Processing image: $nifti_path"

source path/to//conda.sh

conda activate totalsegmentator

echo
echo Location of conda in use:
python -c "import sys; print(sys.executable)"
echo

echo activated:
python -c "import sys; print(sys.executable)"

#Run totalseg
bash /019_run_totalseg.sh "$nifti_path"
