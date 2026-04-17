#!/bin/bash
python trainmultipath_stageone.py \
--name NLST_MultipathGAN_with_anatomy_guidance_Stage_One \
--model resnetmultipathwithoutidentitycycle_gan \
--input_nc 1 \
--output_nc 1 \
--dataset_mode unalignedmultipathstageone \
--batch_size 2 \
--load_size 512 \
--crop_size 512 \
--n_epochs 100 \
--n_epochs_decay 100 \
--gpu_ids 0,1 \
--display_id 0 \
--no_flip \
--norm instance \
--netG_encoder resnet_encoder \
--netG_decoder resnet_decoder \
--netD basic \
--num_threads 30
