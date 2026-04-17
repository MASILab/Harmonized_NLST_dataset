import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.networks import ResBlocklatent
import torch.nn.functional as F 
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np


class ResnetMultipathWithoutIdentityCycleGANModel(BaseModel):
    """
    Utilizes 6 generators and 6 discriminators to perform the cycleGAN task.
    B50f, B30f, BONE, STANDARD, B80f, LUNG
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)') #Forward lambda
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)') #Backward lambda
            parser.add_argument('--lambda_seg', type=float, default=1, help='weight for tissue statistic loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)


        # #For multipath cycleGAN: NEed to include the L2 losses 
        # A - Siemens B50f, B - Siemens B30f, C - GE BONE, D - GE STD, E - GE LUNG, F - Siemens B80f
        self.loss_names = ['D_AB', 'G_AB','cycle_AB', 'segB50fB30f', 'D_BA', 'G_BA','cycle_BA', 'segB30fB50f', 
                           'D_AC', 'G_AC','cycle_AC', 'segB50fBONE', 'D_CA', 'G_CA','cycle_CA', 'segBONEB50f', 
                           'D_AD', 'G_AD','cycle_AD', 'segB50fSTD', 'D_DA', 'G_DA','cycle_DA', 'segSTDB50f', 
                           'D_AE', 'G_AE','cycle_AE', 'segB50fLUNG', 'D_EA', 'G_EA','cycle_EA', 'segLUNGB50f',
                           'D_AF', 'G_AF','cycle_AF', 'segB50fB80f', 'D_FB', 'G_FB','cycle_FB', 'segB80fB50f',
                           'D_BC', 'G_BC','cycle_BC', 'segB30fBONE', 'D_CB', 'G_CB','cycle_CB', 'segBONEB30f',
                           'D_BD', 'G_BD','cycle_BD', 'segB30fSTD', 'D_DB', 'G_DB','cycle_DB', 'segSTDB30f',
                           'D_BE', 'G_BE','cycle_BE', 'segB30fLUNG', 'D_EB', 'G_EB','cycle_EB', 'segLUNGB30f',
                           'D_BF', 'G_BF','cycle_BF', 'segB30fB80f', 'D_FB', 'G_FB','cycle_FB', 'segB80fB30f',
                           'D_CD', 'G_CD','cycle_CD', 'segBONESTD', 'D_DC', 'G_DC','cycle_DC', 'segSTDBONE',
                            'D_CE', 'G_CE','cycle_CE', 'segBONELUNG', 'D_EC', 'G_EC','cycle_EC', 'segLUNGBONE',
                            'D_CF', 'G_CF','cycle_CF', 'segBONEB80f', 'D_FC', 'G_FC','cycle_FC', 'segB80fBONE',
                            'D_DE', 'G_DE','cycle_DE', 'segSTDLUNG', 'D_ED', 'G_ED','cycle_ED', 'segLUNGSTD',
                            'D_DF', 'G_DF','cycle_DF', 'segSTDB80f', 'D_FD', 'G_FD','cycle_FD', 'segB80fSTD',
                            'D_EF', 'G_EF','cycle_EF', 'segLUNGB80f', 'D_FE', 'G_FE','cycle_FE', 'segB80fLUNG',
                        ]
 
        # A = B50f, B = B30f, C = GE BONE, D = GE STD, E = GE LUNG, F = Siemens B80f
        visual_names_AB = ['B50f', 'fake_BA', 'rec_AB'] #Siemens hard to soft
        visual_names_BA = ['B30f', 'fake_AB', 'rec_BA'] #siemens soft to hard
        visual_names_AC = ['B50f', 'fake_CA', 'rec_AC'] #Siemens hard to GE hard
        visual_names_CA = ['BONE', 'fake_AC', 'rec_CA'] #gE hard to Siemens hard
        visual_names_AD = ['B50f', 'fake_DA', 'rec_AD'] #Siemens hard to GE soft
        visual_names_DA = ['STD', 'fake_AD', 'rec_DA'] #GE soft to Siemens hard
        visual_names_AE = ['B50f', 'fake_EA', 'rec_AE'] #Siemens hard to GE LUNG
        visual_names_EA = ['LUNG', 'fake_AE', 'rec_EA'] #GE LUNG to Siemens hard
        visual_names_AF = ['B50f', 'fake_FA', 'rec_AF'] #Siemens hard to Siemens B80f
        visual_names_FA = ['B80f', 'fake_AF', 'rec_FA'] #Siemens B80f to Siemens hard
        visual_names_BC = ['B30f', 'fake_CB', 'rec_BC'] #Siemens soft GE hard
        visual_names_CB = ['BONE', 'fake_BC', 'rec_CB'] #GE hard siemens soft
        visual_names_BD = ['B30f', 'fake_DB', 'rec_BD'] #Siemens soft GE soft
        visual_names_DB = ['STD', 'fake_BD', 'rec_DB'] #GE soft Siemens soft
        visual_names_BE = ['B30f', 'fake_EB', 'rec_BE'] #Siemens soft GE LUNG
        visual_names_EB = ['LUNG', 'fake_BE', 'rec_EB'] #GE LUNG Siemens soft
        visual_names_BF = ['B30f', 'fake_FB', 'rec_BF'] #Siemens soft Siemens B80f
        visual_names_FB = ['B80f', 'fake_BF', 'rec_FB'] #Siemens B80f Siemens soft
        visual_names_CD = ['BONE', 'fake_DC', 'rec_CD'] #GE hard GE soft
        visual_names_DC = ['STD', 'fake_CD', 'rec_DC'] #GE soft GE hard
        visual_names_CE = ['BONE', 'fake_EC', 'rec_CE'] #GE hard GE LUNG
        visual_names_EC = ['LUNG', 'fake_CE', 'rec_EC'] #GE LUNG GE hard
        visual_names_CF = ['BONE', 'fake_FC', 'rec_CF'] #GE hard Siemens B80f
        visual_names_FC = ['B80f', 'fake_CF', 'rec_FC'] #Siemens B80f GE hard
        visual_names_DE = ['STD', 'fake_ED', 'rec_DE'] #GE soft GE LUNG
        visual_names_ED = ['LUNG', 'fake_DE', 'rec_ED'] #GE LUNG GE soft
        visual_names_DF = ['STD', 'fake_FD', 'rec_DF'] #GE soft Siemens B80f
        visual_names_FD = ['B80f', 'fake_DF', 'rec_FD'] #Siemens B80f GE soft
        visual_names_EF = ['LUNG', 'fake_FE', 'rec_EF'] #GE LUNG Siemens B80f
        visual_names_FE = ['B80f', 'fake_EF', 'rec_FE'] #Siemens B80f GE LUNG


        #Multipath GAN
        self.visual_names = visual_names_AB + visual_names_BA + visual_names_AC + visual_names_CA + visual_names_AD + visual_names_DA + visual_names_AE + visual_names_EA + \
                            visual_names_AF + visual_names_FA + visual_names_BC + visual_names_CB + visual_names_BD + visual_names_DB + visual_names_BE + visual_names_EB + \
                            visual_names_BF + visual_names_FB + visual_names_CD + visual_names_DC + visual_names_CE + visual_names_EC + visual_names_CF + visual_names_FC + \
                            visual_names_DE + visual_names_ED + visual_names_DF + visual_names_FD + visual_names_EF + visual_names_FE
                         


        if self.isTrain:
            self.model_names = ['G_B50f_encoder', 'G_B50f_decoder', 'G_B30f_encoder', 'G_B30f_decoder',
                                'G_BONE_encoder', 'G_BONE_decoder','G_STD_encoder', 'G_STD_decoder',
                                'G_LUNG_encoder', 'G_LUNG_decoder','G_B80f_encoder', 'G_B80f_decoder',
                                'D_A', 'D_B', 'D_C', 'D_D', 'D_E', 'D_F',]
        else:  # during test time, only load Gs
            self.model_names = ['G_B50f_encoder', 'G_B50f_decoder', 'G_B30f_encoder', 'G_B30f_decoder',
                                'G_BONE_decoder', 'G_BONE_encoder','G_STD_decoder', 'G_STD_encoder',
                                'G_LUNG_decoder', 'G_LUNG_encoder','G_B80f_decoder', 'G_B80f_encoder']


        shared_latent = ResBlocklatent(n_blocks=9, ngf=64, norm_layer=nn.InstanceNorm2d, padding_type='reflect').to(self.device)
        #Multipath cycleGAN encoder-decoder initalizations
        self.netG_B50f_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #B50f encoder
        self.netG_B50f_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #B50f decoder
        
        self.netG_B30f_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #B30f encoder
        self.netG_B30f_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #B30f decoder
        
        self.netG_BONE_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #BONE encoder
        self.netG_BONE_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #BONE decoder
        
        self.netG_STD_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #STD encoder
        self.netG_STD_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #STD decoder
        
        self.netG_LUNG_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #LUNG encoder
        self.netG_LUNG_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #LUNG decoder

        self.netG_B80f_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #B80f encoder
        self.netG_B80f_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #B80f decoder

        
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # Siemens B50f
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # Siemens B30f
            self.netD_C = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # GE BONE
            self.netD_D = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # GE STANDARD
            self.netD_E = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # GE LUNG
            self.netD_F = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # Siemens B80f


        if self.isTrain:

            self.fake_AB_pool = ImagePool(opt.pool_size) #Siemens soft to Siemens hard (B to A)
            self.fake_BA_pool = ImagePool(opt.pool_size) #Siemens hard to Siemens soft (A to B)
            self.fake_AC_pool = ImagePool(opt.pool_size) #GE hard to Siemens hard (C to A)
            self.fake_CA_pool = ImagePool(opt.pool_size) #Siemens hard to GE hard (A to C) 
            self.fake_AD_pool = ImagePool(opt.pool_size) #GE soft to Siemens hard (D to A)
            self.fake_DA_pool = ImagePool(opt.pool_size) #Siemens hard to GE soft (A to D)
            self.fake_AE_pool = ImagePool(opt.pool_size) #GE LUNG to Siemens hard (E to A)
            self.fake_EA_pool = ImagePool(opt.pool_size) #Siemens hard to GE LUNG (A to E)
            self.fake_AF_pool = ImagePool(opt.pool_size) #Siemens B80f to Siemens hard (F to A)
            self.fake_FA_pool = ImagePool(opt.pool_size) #Siemens hard to Siemens B80f (A to F)
            
            self.fake_BC_pool = ImagePool(opt.pool_size) #GE hard to Siemens soft (C to B)
            self.fake_CB_pool = ImagePool(opt.pool_size) #Siemens soft to GE hard (B to C)
            self.fake_BD_pool = ImagePool(opt.pool_size) #GE soft to siemens soft (D to B)
            self.fake_DB_pool = ImagePool(opt.pool_size) #Siemens soft to GE soft (B to D)
            self.fake_BE_pool = ImagePool(opt.pool_size) #GE LUNG to Siemens soft (E to B)
            self.fake_EB_pool = ImagePool(opt.pool_size) #Siemens soft to GE LUNG (B to E)
            self.fake_BF_pool = ImagePool(opt.pool_size) #Siemens B80f to Siemens soft (F to B)
            self.fake_FB_pool = ImagePool(opt.pool_size) #Siemens soft to Siemens B80f (B to F)

            self.fake_CD_pool = ImagePool(opt.pool_size) #GE hard to GE soft (C to D)
            self.fake_DC_pool = ImagePool(opt.pool_size) #GE soft to GE hard (D to C)
            self.fake_CE_pool = ImagePool(opt.pool_size) #GE LUNG to GE hard (E to C)
            self.fake_EC_pool = ImagePool(opt.pool_size) #GE hard to GE LUNG (C to E)
            self.fake_CF_pool = ImagePool(opt.pool_size) #Siemens B80f to GE hard (F to C)
            self.fake_FC_pool = ImagePool(opt.pool_size) #GE hard to Siemens B80f (C to F)
            
            self.fake_DE_pool = ImagePool(opt.pool_size) #GE soft to GE LUNG (E to D)
            self.fake_ED_pool = ImagePool(opt.pool_size) #GE LUNG to GE soft (D to E)
            self.fake_DF_pool = ImagePool(opt.pool_size) #Siemens B80f to GE soft (F to D)
            self.fake_FD_pool = ImagePool(opt.pool_size) #GE soft to Siemens B80f (D to F)

            self.fake_EF_pool = ImagePool(opt.pool_size) #Siemens B80f to GE LUNG (F to E)
            self.fake_FE_pool = ImagePool(opt.pool_size) #GE LUNG to Siemens B80f (E to F)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss() #cycle loss
            self.criterionSeg = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B50f_encoder.parameters(),
                                                                self.netG_B50f_decoder.parameters(),
                                                                self.netG_B30f_encoder.parameters(),
                                                                self.netG_B30f_decoder.parameters(),
                                                                self.netG_BONE_encoder.parameters(),
                                                                self.netG_BONE_decoder.parameters(),
                                                                self.netG_STD_encoder.parameters(),
                                                                self.netG_STD_decoder.parameters(),
                                                                self.netG_LUNG_encoder.parameters(),
                                                                self.netG_LUNG_decoder.parameters(),
                                                                self.netG_B80f_encoder.parameters(),
                                                                self.netG_B80f_decoder.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),
                                                                self.netD_B.parameters(),
                                                                self.netD_C.parameters(),
                                                                self.netD_D.parameters(),
                                                                self.netD_E.parameters(),
                                                                self.netD_F.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            print(self.optimizer_G.state_dict())
            print(self.optimizer_D.state_dict())
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.scalar = GradScaler()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # Multipath GAN (Single GPU)
        self.B50f = input['B50f_image'].to(self.device)
        self.B30f = input['B30f_image'].to(self.device) 
        self.BONE = input['BONE_image'].to(self.device)
        self.STD = input['STANDARD_image'].to(self.device)
        self.B80f = input['B80f_image'].to(self.device)
        self.LUNG = input['LUNG_image'].to(self.device)

        self.B50f_mask = input['B50f_mask'].to(self.device)
        self.B30f_mask = input['B30f_mask'].to(self.device)
        self.BONE_mask = input['BONE_mask'].to(self.device)
        self.STD_mask = input['STANDARD_mask'].to(self.device)
        self.LUNG_mask = input['LUNG_mask'].to(self.device)
        self.B80f_mask = input['B80f_mask'].to(self.device)


     
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def tissue_statistic_loss(self, real_A, fake_A, real_B, fake_B, real_mask_A, real_mask_B):
        unique_labels_forward = torch.unique(real_mask_A)
        unique_labels_forward = unique_labels_forward[unique_labels_forward != 0]
        unique_labels_backward = torch.unique(real_mask_B)
        unique_labels_backward = unique_labels_backward[unique_labels_backward != 0]

        real_mean_A = torch.zeros(len(unique_labels_forward))
        fake_mean_B = torch.zeros(len(unique_labels_forward))
        real_mean_B = torch.zeros(len(unique_labels_backward))
        fake_mean_A = torch.zeros(len(unique_labels_backward))

        for i, label in enumerate(unique_labels_forward): #elementwise iteration and mean compuation
            if torch.sum(real_mask_A == label) > 5: #(This only penalizes the labels greater than label number 5. The logic needs to be changed)
                real_mean_A[i] = torch.mean(real_A[real_mask_A == label])
                fake_mean_B[i] = torch.mean(fake_B[real_mask_A == label])
            else: #Less than 5 pixels in the mask for a given label
                real_mean_A[i] = 0.0
                fake_mean_B[i] = 0.0
        
        for i, label in enumerate(unique_labels_backward):
            if torch.sum(real_mask_B == label) > 5:
                real_mean_B[i] = torch.mean(real_B[real_mask_B == label])
                fake_mean_A[i] = torch.mean(fake_A[real_mask_B == label])
            else:
                real_mean_B[i] = 0.0
                fake_mean_A[i] = 0.0
        
        loss_seg_A = self.criterionSeg(real_mean_A, fake_mean_B) * self.opt.lambda_seg
        loss_seg_B = self.criterionSeg(real_mean_B, fake_mean_A) * self.opt.lambda_seg

        return loss_seg_A, loss_seg_B

    def cyclicpath(self, latent, target_decoder, target_encoder, source_decoder):
        """
        Code snippet for a given cyclic path in the multipath GAN model
        """
        fake_image = target_decoder(latent) #Decoding latent to fake image
        rec_latent = target_encoder(fake_image) #Encoding fake image to latent
        rec_image = source_decoder(rec_latent) #Decoding latent to reconstructed image
        return fake_image, rec_image #Returning fake image and reconstructed image
        
    

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with autocast():
            latent_B50f = self.netG_B50f_encoder(self.B50f) #A
            latent_B30f = self.netG_B30f_encoder(self.B30f) #B
            latent_BONE = self.netG_BONE_encoder(self.BONE) #C
            latent_STD = self.netG_STD_encoder(self.STD) #D
            latent_LUNG = self.netG_LUNG_encoder(self.LUNG) #E 
            latent_B80f = self.netG_B80f_encoder(self.B80f) #F

            #Siemens B50f
            #Generator 1: Siemens B50f -> B30f
            self.fake_BA, self.rec_AB = self.cyclicpath(latent_B50f, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_B50f_decoder) 
            #Generator 2: Siemens B30f -> B50f
            self.fake_AB, self.rec_BA = self.cyclicpath(latent_B30f, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_B30f_decoder) 
            #Generator 3: Siemens B50f -> GE BONE 
            self.fake_CA, self.rec_AC = self.cyclicpath(latent_B50f, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_B50f_decoder) 
            #Generator 4: GE BONE -> Siemens B50f
            self.fake_AC, self.rec_CA = self.cyclicpath(latent_BONE, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_BONE_decoder) 
            #Generator 5: Siemens B50f -> GE STD
            self.fake_DA, self.rec_AD = self.cyclicpath(latent_B50f, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_B50f_decoder) 
            #Generator 6: GE STD -> Siemens B50f
            self.fake_AD, self.rec_DA = self.cyclicpath(latent_STD, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_STD_decoder) 
            #Generator 7: Siemens B50f -> GE LUNG
            self.fake_EA, self.rec_AE = self.cyclicpath(latent_B50f, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_B50f_decoder) 
            #Generator 8: GE LUNG -> Siemens B50f
            self.fake_AE, self.rec_EA = self.cyclicpath(latent_LUNG, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_LUNG_decoder) 
            #Generator 9: Siemens B50f -> Siemens B80f
            self.fake_FA, self.rec_AF = self.cyclicpath(latent_B50f, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_B50f_decoder) 
            #Generator 10: Siemens B80f -> Siemens B50f
            self.fake_AF, self.rec_FA = self.cyclicpath(latent_B80f, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_B80f_decoder) 
           
            #Siemens B30f
            #Generator 11: Siemens B30f -> GE BONE
            self.fake_CB, self.rec_BC = self.cyclicpath(latent_B30f, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_B30f_decoder) 
            #Generator 12: GE BONE -> Siemens B30f
            self.fake_BC, self.rec_CB = self.cyclicpath(latent_BONE, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_BONE_decoder) 
            #Generator 13: Siemens B30f -> GE STD
            self.fake_DB, self.rec_BD = self.cyclicpath(latent_B30f, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_B30f_decoder) 
            #Generator 14: GE STD -> Siemens B30f
            self.fake_BD, self.rec_DB = self.cyclicpath(latent_STD, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_STD_decoder) 
            #Generator 15: Siemens B30f -> GE LUNG
            self.fake_EB, self.rec_BE = self.cyclicpath(latent_B30f, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_B30f_decoder) 
            #Generator 16: GE LUNG -> Siemens B30f
            self.fake_BE, self.rec_EB = self.cyclicpath(latent_LUNG, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_LUNG_decoder) 
            #Generator 17: Siemens B30f -> Siemens B80f
            self.fake_FB, self.rec_BF = self.cyclicpath(latent_B30f, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_B30f_decoder) 
            #Generator 18: Siemens B80f -> Siemens B30f
            self.fake_BF, self.rec_FB = self.cyclicpath(latent_B80f, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_B80f_decoder) 

            #GE BONE
            #Generator 19: GE BONE -> GE STD
            self.fake_DC, self.rec_CD = self.cyclicpath(latent_BONE, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_BONE_decoder) 
            #Generator 20: GE STD -> GE BONE
            self.fake_CD, self.rec_DC = self.cyclicpath(latent_STD, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_STD_decoder) 
            #Generator 21: GE BONE -> GE LUNG
            self.fake_EC, self.rec_CE = self.cyclicpath(latent_BONE, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_BONE_decoder) 
            #Generator 22: GE LUNG -> GE BONE
            self.fake_CE, self.rec_EC = self.cyclicpath(latent_LUNG, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_LUNG_decoder) 
            #Generator 23: GE BONE -> Siemens B80f
            self.fake_FC, self.rec_CF = self.cyclicpath(latent_BONE, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_BONE_decoder) 
            #Generator 24: Siemens B80f -> GE BONE
            self.fake_CF, self.rec_FC = self.cyclicpath(latent_B80f, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_B80f_decoder)

            #GE STD
            #Generator 25: GE STD -> GE LUNG
            self.fake_ED, self.rec_DE = self.cyclicpath(latent_STD, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_STD_decoder) 
            #Generator 26: GE LUNG -> GE STD
            self.fake_DE, self.rec_ED = self.cyclicpath(latent_LUNG, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_LUNG_decoder) 
            #Generator 27: GE STD -> Siemens B80f
            self.fake_FD, self.rec_DF = self.cyclicpath(latent_STD, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_STD_decoder) 
            #Generator 28: Siemens B80f -> GE STD
            self.fake_DF, self.rec_FD = self.cyclicpath(latent_B80f, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_B80f_decoder)

            #GE LUNG
            #Generator 29: GE LUNG -> Siemens B80f
            self.fake_FE, self.rec_EF = self.cyclicpath(latent_LUNG, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_LUNG_decoder) 
            #Generator 30: Siemens B80f -> GE LUNG
            self.fake_EF, self.rec_FE = self.cyclicpath(latent_B80f, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_B80f_decoder)


    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
       
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
    
    # Mulitpath cycleGAN
    def backward_D(self):
        #Siemens B50f
        """Calculate GAN loss for discriminator D_AB (B50f to B30f)"""
        fake_BA = self.fake_BA_pool.query(self.fake_BA)
        self.loss_D_AB = self.backward_D_basic(self.netD_A, self.B30f, fake_BA)

        """Calculate GAN loss for discriminator D_BA (B30f to B50f)"""
        fake_AB = self.fake_AB_pool.query(self.fake_AB)
        self.loss_D_BA = self.backward_D_basic(self.netD_B, self.B50f, fake_AB)

        """Calculate GAN loss for discriminator D_AC (B50f to BONE)"""
        fake_CA = self.fake_CA_pool.query(self.fake_CA)
        self.loss_D_AC = self.backward_D_basic(self.netD_A, self.BONE, fake_CA)

        """Calculate GAN loss for discriminator D_CA (BONE to B50f)"""
        fake_AC = self.fake_AC_pool.query(self.fake_AC)
        self.loss_D_CA = self.backward_D_basic(self.netD_C, self.B50f, fake_AC)

        """Calculate GAN loss for discriminator D_AD (B50f to STD)"""
        fake_DA = self.fake_DA_pool.query(self.fake_DA)
        self.loss_D_AD = self.backward_D_basic(self.netD_A, self.STD, fake_DA)

        """Calculate GAN loss for discriminator D_DA (STD to B50f)"""
        fake_AD = self.fake_AD_pool.query(self.fake_AD)
        self.loss_D_DA = self.backward_D_basic(self.netD_D, self.B50f, fake_AD)

        """Calculate GAN loss for discriminator D_AE (B50f to LUNG)"""
        fake_EA = self.fake_EA_pool.query(self.fake_EA)
        self.loss_D_AE = self.backward_D_basic(self.netD_A, self.LUNG, fake_EA)

        """Calculate GAN loss for discriminator D_EA (LUNG to B50f)"""
        fake_AE = self.fake_AE_pool.query(self.fake_AE)
        self.loss_D_EA = self.backward_D_basic(self.netD_E, self.B50f, fake_AE)

        """Calculate GAN loss for discriminator D_AF (B50f to B80f)"""
        fake_FA = self.fake_FA_pool.query(self.fake_FA)
        self.loss_D_AF = self.backward_D_basic(self.netD_A, self.B80f, fake_FA)

        """Calculate GAN loss for discriminator D_FA (B80f to B50f)"""
        fake_AF = self.fake_AF_pool.query(self.fake_AF)
        self.loss_D_FA = self.backward_D_basic(self.netD_F, self.B50f, fake_AF)

       #Siemens B30f
        """Calculate GAN loss for discriminator D_BC (B30f to BONE)"""
        fake_CB = self.fake_CB_pool.query(self.fake_CB)
        self.loss_D_BC = self.backward_D_basic(self.netD_B, self.BONE, fake_CB)

        """Calculate GAN loss for discriminator D_CB (BONE to B30f)"""
        fake_BC = self.fake_BC_pool.query(self.fake_BC)
        self.loss_D_CB = self.backward_D_basic(self.netD_C, self.B30f, fake_BC)

        """Calculate GAN loss for discriminator D_BD (B30f to STD)"""
        fake_DB = self.fake_DB_pool.query(self.fake_DB)
        self.loss_D_BD = self.backward_D_basic(self.netD_B, self.STD, fake_DB)

        """Calculate GAN loss for discriminator D_DB (STD to B30f)"""
        fake_BD = self.fake_BD_pool.query(self.fake_BD)
        self.loss_D_DB = self.backward_D_basic(self.netD_D, self.B30f, fake_BD)

        """Calculate GAN loss for discriminator D_BE (B30f to LUNG)"""
        fake_EB = self.fake_EB_pool.query(self.fake_EB)
        self.loss_D_BE = self.backward_D_basic(self.netD_B, self.LUNG, fake_EB)

        """Calculate GAN loss for discriminator D_EB (LUNG to B30f)"""
        fake_BE = self.fake_BE_pool.query(self.fake_BE)
        self.loss_D_EB = self.backward_D_basic(self.netD_E, self.B30f, fake_BE)

        """Calculate GAN loss for discriminator D_BF (B30f to B80f)"""
        fake_FB = self.fake_FB_pool.query(self.fake_FB)
        self.loss_D_BF = self.backward_D_basic(self.netD_B, self.B80f, fake_FB)

        """Calculate GAN loss for discriminator D_FB (B80f to B30f)"""
        fake_BF = self.fake_BF_pool.query(self.fake_BF)
        self.loss_D_FB = self.backward_D_basic(self.netD_F, self.B30f, fake_BF)

        #GE BONE
        """Calculate GAN loss for discriminator D_CD (BONE to STD)"""
        fake_DC = self.fake_DC_pool.query(self.fake_DC)
        self.loss_D_CD = self.backward_D_basic(self.netD_C, self.STD, fake_DC)

        """Calculate GAN loss for discriminator D_CB (STD to BONE)"""
        fake_CD = self.fake_CD_pool.query(self.fake_CD)
        self.loss_D_DC = self.backward_D_basic(self.netD_D, self.BONE, fake_CD)

        """Calculate GAN loss for discriminator D_CE (BONE to LUNG)"""
        fake_EC = self.fake_EC_pool.query(self.fake_EC)
        self.loss_D_CE = self.backward_D_basic(self.netD_C, self.LUNG, fake_EC)

        """Calculate GAN loss for discriminator D_EC (LUNG to BONE)"""
        fake_CE = self.fake_CE_pool.query(self.fake_CE)
        self.loss_D_EC = self.backward_D_basic(self.netD_E, self.BONE, fake_CE)

        """Calculate GAN loss for discriminator D_CF (BONE to B80f)"""
        fake_FC = self.fake_FC_pool.query(self.fake_FC)
        self.loss_D_CF = self.backward_D_basic(self.netD_C, self.B80f, fake_FC)

        """Calculate GAN loss for discriminator D_FC (B80f to BONE)"""
        fake_CF = self.fake_CF_pool.query(self.fake_CF)
        self.loss_D_FC = self.backward_D_basic(self.netD_F, self.BONE, fake_CF)

        #GE STD
        """Calculate GAN loss for discriminator D_DE (STD to LUNG)"""
        fake_ED = self.fake_ED_pool.query(self.fake_ED)
        self.loss_D_DE = self.backward_D_basic(self.netD_D, self.LUNG, fake_ED)

        """Calculate GAN loss for discriminator D_ED (LUNG to STD)"""
        fake_DE = self.fake_DE_pool.query(self.fake_DE)
        self.loss_D_ED = self.backward_D_basic(self.netD_E, self.STD, fake_DE)

        """Calculate GAN loss for discriminator D_DF (STD to B80f)"""
        fake_FD = self.fake_FD_pool.query(self.fake_FD)
        self.loss_D_DF = self.backward_D_basic(self.netD_D, self.B80f, fake_FD)

        """Calculate GAN loss for discriminator D_FD (B80f to STD)"""
        fake_DF = self.fake_DF_pool.query(self.fake_DF)
        self.loss_D_FD = self.backward_D_basic(self.netD_F, self.STD, fake_DF)

        #GE LUNG
        """Calculate GAN loss for discriminator D_EF (LUNG to B80f)"""
        fake_FE = self.fake_FE_pool.query(self.fake_FE)
        self.loss_D_EF = self.backward_D_basic(self.netD_E, self.B80f, fake_FE)

        """Calculate GAN loss for discriminator D_FE (B80f to LUNG)"""
        fake_EF = self.fake_EF_pool.query(self.fake_EF)
        self.loss_D_FE = self.backward_D_basic(self.netD_F, self.LUNG, fake_EF)

        self.loss_D = self.loss_D_AB + self.loss_D_BA + self.loss_D_AC + self.loss_D_CA + self.loss_D_AD + self.loss_D_DA + self.loss_D_AE + self.loss_D_EA + \
                      self.loss_D_AF + self.loss_D_FA + self.loss_D_BC + self.loss_D_CB + self.loss_D_BD + self.loss_D_DB + self.loss_D_BE + self.loss_D_EB + \
                      self.loss_D_BF + self.loss_D_FB + self.loss_D_CD + self.loss_D_DC + self.loss_D_CE + self.loss_D_EC + self.loss_D_CF + self.loss_D_FC + \
                      self.loss_D_DE + self.loss_D_ED + self.loss_D_DF + self.loss_D_FD + self.loss_D_EF + self.loss_D_FE
                       

        return self.loss_D

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
    
 
        # Least squares loss for all the generators
        #B50f
        self.loss_G_AB = self.criterionGAN(self.netD_A(self.fake_BA), True)
        self.loss_G_BA = self.criterionGAN(self.netD_B(self.fake_AB), True)
        self.loss_G_AC = self.criterionGAN(self.netD_A(self.fake_CA), True)
        self.loss_G_CA = self.criterionGAN(self.netD_C(self.fake_AC), True)
        self.loss_G_AD = self.criterionGAN(self.netD_A(self.fake_DA), True)
        self.loss_G_DA = self.criterionGAN(self.netD_D(self.fake_AD), True)
        self.loss_G_AE = self.criterionGAN(self.netD_A(self.fake_EA), True)
        self.loss_G_EA = self.criterionGAN(self.netD_E(self.fake_AE), True)
        self.loss_G_AF = self.criterionGAN(self.netD_A(self.fake_FA), True)
        self.loss_G_FA = self.criterionGAN(self.netD_F(self.fake_AF), True)

        #B30f
        self.loss_G_BC = self.criterionGAN(self.netD_B(self.fake_CB), True)
        self.loss_G_CB = self.criterionGAN(self.netD_C(self.fake_BC), True)
        self.loss_G_BD = self.criterionGAN(self.netD_B(self.fake_DB), True)
        self.loss_G_DB = self.criterionGAN(self.netD_D(self.fake_BD), True)
        self.loss_G_BE = self.criterionGAN(self.netD_B(self.fake_EB), True)
        self.loss_G_EB = self.criterionGAN(self.netD_E(self.fake_BE), True)
        self.loss_G_BF = self.criterionGAN(self.netD_B(self.fake_FB), True)
        self.loss_G_FB = self.criterionGAN(self.netD_F(self.fake_BF), True)

        #GE BONE
        self.loss_G_CD = self.criterionGAN(self.netD_C(self.fake_DC), True)
        self.loss_G_DC = self.criterionGAN(self.netD_D(self.fake_CD), True)
        self.loss_G_CE = self.criterionGAN(self.netD_C(self.fake_EC), True)
        self.loss_G_EC = self.criterionGAN(self.netD_E(self.fake_CE), True)
        self.loss_G_CF = self.criterionGAN(self.netD_C(self.fake_FC), True)
        self.loss_G_FC = self.criterionGAN(self.netD_F(self.fake_CF), True)

        #GE STD
        self.loss_G_DE = self.criterionGAN(self.netD_D(self.fake_ED), True)
        self.loss_G_ED = self.criterionGAN(self.netD_E(self.fake_DE), True)
        self.loss_G_DF = self.criterionGAN(self.netD_D(self.fake_FD), True)
        self.loss_G_FD = self.criterionGAN(self.netD_F(self.fake_DF), True)

        #GE LUNG
        self.loss_G_EF = self.criterionGAN(self.netD_E(self.fake_FE), True)
        self.loss_G_FE = self.criterionGAN(self.netD_F(self.fake_EF), True)
 

        #Multipath cycleGAN: Cycle consistency losses
        #B50f
        self.loss_cycle_AB = self.criterionCycle(self.rec_AB, self.B50f) * lambda_A
        self.loss_cycle_BA = self.criterionCycle(self.rec_BA, self.B30f) * lambda_B
        self.loss_cycle_AC = self.criterionCycle(self.rec_AC, self.B50f) * lambda_A
        self.loss_cycle_CA = self.criterionCycle(self.rec_CA, self.BONE) * lambda_B
        self.loss_cycle_AD = self.criterionCycle(self.rec_AD, self.B50f) * lambda_A
        self.loss_cycle_DA = self.criterionCycle(self.rec_DA, self.STD) * lambda_B
        self.loss_cycle_AE = self.criterionCycle(self.rec_AE, self.B50f) * lambda_A
        self.loss_cycle_EA = self.criterionCycle(self.rec_EA, self.LUNG) * lambda_B
        self.loss_cycle_AF = self.criterionCycle(self.rec_AF, self.B50f) * lambda_A
        self.loss_cycle_FA = self.criterionCycle(self.rec_FA, self.B80f) * lambda_B

        #B30f
        self.loss_cycle_BC = self.criterionCycle(self.rec_BC, self.B30f) * lambda_A
        self.loss_cycle_CB = self.criterionCycle(self.rec_CB, self.BONE) * lambda_B
        self.loss_cycle_BD = self.criterionCycle(self.rec_BD, self.B30f) * lambda_A
        self.loss_cycle_DB = self.criterionCycle(self.rec_DB, self.STD) * lambda_B
        self.loss_cycle_BE = self.criterionCycle(self.rec_BE, self.B30f) * lambda_A
        self.loss_cycle_EB = self.criterionCycle(self.rec_EB, self.LUNG) * lambda_B
        self.loss_cycle_BF = self.criterionCycle(self.rec_BF, self.B30f) * lambda_A
        self.loss_cycle_FB = self.criterionCycle(self.rec_FB, self.B80f) * lambda_B

        #BONE
        self.loss_cycle_CD = self.criterionCycle(self.rec_CD, self.BONE) * lambda_A
        self.loss_cycle_DC = self.criterionCycle(self.rec_DC, self.STD) * lambda_B
        self.loss_cycle_CE = self.criterionCycle(self.rec_CE, self.BONE) * lambda_A
        self.loss_cycle_EC = self.criterionCycle(self.rec_EC, self.LUNG) * lambda_B
        self.loss_cycle_CF = self.criterionCycle(self.rec_CF, self.BONE) * lambda_A
        self.loss_cycle_FC = self.criterionCycle(self.rec_FC, self.B80f) * lambda_B

        #STD
        self.loss_cycle_DE = self.criterionCycle(self.rec_DE, self.STD) * lambda_A
        self.loss_cycle_ED = self.criterionCycle(self.rec_ED, self.LUNG) * lambda_B
        self.loss_cycle_DF = self.criterionCycle(self.rec_DF, self.STD) * lambda_A
        self.loss_cycle_FD = self.criterionCycle(self.rec_FD, self.B80f) * lambda_B

        #LUNG
        self.loss_cycle_EF = self.criterionCycle(self.rec_EF, self.LUNG) * lambda_A
        self.loss_cycle_FE = self.criterionCycle(self.rec_FE, self.B80f) * lambda_B
       
        
        #Tissue statistic loss
        #B50f
        self.loss_segB50fB30f, self.loss_segB30fB50f = self.tissue_statistic_loss(self.B50f, self.fake_AB, self.B30f, self.fake_BA, self.B50f_mask, self.B30f_mask)
        self.loss_segB50fBONE, self.loss_segBONEB50f = self.tissue_statistic_loss(self.B50f, self.fake_AC, self.BONE, self.fake_CA, self.B50f_mask, self.BONE_mask)
        self.loss_segB50fSTD, self.loss_segSTDB50f = self.tissue_statistic_loss(self.B50f, self.fake_AD, self.STD, self.fake_DA, self.B50f_mask, self.STD_mask)
        self.loss_segB50fLUNG, self.loss_segLUNGB50f = self.tissue_statistic_loss(self.B50f, self.fake_AE, self.LUNG, self.fake_EA, self.B50f_mask, self.LUNG_mask)
        self.loss_segB50fB80f, self.loss_segB80fB50f = self.tissue_statistic_loss(self.B50f, self.fake_AF, self.B80f, self.fake_FA, self.B50f_mask, self.B80f_mask)

        #B30f 
        self.loss_segB30fBONE, self.loss_segBONEB30f = self.tissue_statistic_loss(self.B30f, self.fake_BC, self.BONE, self.fake_CB, self.B30f_mask, self.BONE_mask)
        self.loss_segB30fSTD, self.loss_segSTDB30f = self.tissue_statistic_loss(self.B30f, self.fake_BD, self.STD, self.fake_DB, self.B30f_mask, self.STD_mask)
        self.loss_segB30fLUNG, self.loss_segLUNGB30f = self.tissue_statistic_loss(self.B30f, self.fake_BE, self.LUNG, self.fake_EB, self.B30f_mask, self.LUNG_mask)
        self.loss_segB30fB80f, self.loss_segB80fB30f = self.tissue_statistic_loss(self.B30f, self.fake_BF, self.B80f, self.fake_FB, self.B30f_mask, self.B80f_mask)

        #BONE
        self.loss_segBONESTD, self.loss_segSTDBONE = self.tissue_statistic_loss(self.BONE, self.fake_CD, self.STD, self.fake_DC, self.BONE_mask, self.STD_mask)
        self.loss_segBONELUNG, self.loss_segLUNGBONE = self.tissue_statistic_loss(self.BONE, self.fake_CE, self.LUNG, self.fake_EC, self.BONE_mask, self.LUNG_mask)
        self.loss_segBONEB80f, self.loss_segB80fBONE = self.tissue_statistic_loss(self.BONE, self.fake_CF, self.B80f, self.fake_FC, self.BONE_mask, self.B80f_mask)

        #STD
        self.loss_segSTDLUNG, self.loss_segLUNGSTD = self.tissue_statistic_loss(self.STD, self.fake_DE, self.LUNG, self.fake_ED, self.STD_mask, self.LUNG_mask)
        self.loss_segSTDB80f, self.loss_segB80fSTD = self.tissue_statistic_loss(self.STD, self.fake_DF, self.B80f, self.fake_FD, self.STD_mask, self.B80f_mask)

        #LUNG
        self.loss_segLUNGB80f, self.loss_segB80fLUNG = self.tissue_statistic_loss(self.LUNG, self.fake_EF, self.B80f, self.fake_FE, self.LUNG_mask, self.B80f_mask)
 
       
        #this is loss function for multipath cycleGAN: Adversarial losses + desicriminator losses + Seg losses
        self.loss_G =  self.loss_G_AB + self.loss_G_BA + self.loss_cycle_AB + self.loss_cycle_BA + self.loss_G_AC + self.loss_G_CA + self.loss_cycle_AC + self.loss_cycle_CA + \
                       self.loss_G_AD + self.loss_G_DA + self.loss_cycle_AD + self.loss_cycle_DA + self.loss_G_AE + self.loss_G_EA + self.loss_cycle_AE + self.loss_cycle_EA + \
                       self.loss_G_AF + self.loss_G_FA + self.loss_cycle_AF + self.loss_cycle_FA + self.loss_G_BC + self.loss_G_CB + self.loss_cycle_BC + self.loss_cycle_CB + \
                       self.loss_G_BD + self.loss_G_DB + self.loss_cycle_BD + self.loss_cycle_DB + self.loss_G_BE + self.loss_G_EB + self.loss_cycle_BE + self.loss_cycle_EB + \
                       self.loss_G_BF + self.loss_G_FB + self.loss_cycle_BF + self.loss_cycle_FB + self.loss_G_CD + self.loss_G_DC + self.loss_cycle_CD + self.loss_cycle_DC + \
                       self.loss_G_CE + self.loss_G_EC + self.loss_cycle_CE + self.loss_cycle_EC + self.loss_G_CF + self.loss_G_FC + self.loss_cycle_CF + self.loss_cycle_FC + \
                       self.loss_G_DE + self.loss_G_ED + self.loss_cycle_DE + self.loss_cycle_ED + self.loss_G_DF + self.loss_G_FD + self.loss_cycle_DF + self.loss_cycle_FD + \
                       self.loss_G_EF + self.loss_G_FE + self.loss_cycle_EF + self.loss_cycle_FE + self.loss_segB50fB30f + self.loss_segB30fB50f + self.loss_segB50fBONE + self.loss_segBONEB50f + \
                       self.loss_segB50fSTD + self.loss_segSTDB50f + self.loss_segB50fB80f + self.loss_segB80fB50f + self.loss_segB50fLUNG + self.loss_segLUNGB50f + \
                       self.loss_segB30fBONE + self.loss_segBONEB30f + self.loss_segB30fSTD + self.loss_segSTDB30f + self.loss_segB30fB80f + self.loss_segB80fB30f + \
                       self.loss_segB30fLUNG + self.loss_segLUNGB30f + self.loss_segBONESTD + self.loss_segSTDBONE + self.loss_segBONELUNG + self.loss_segLUNGBONE + \
                       self.loss_segBONEB80f + self.loss_segB80fBONE + self.loss_segSTDLUNG + self.loss_segLUNGSTD + self.loss_segSTDB80f + self.loss_segB80fSTD + \
                       self.loss_segLUNGB80f + self.loss_segB80fLUNG

        return self.loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F], False) #Multipath 
        self.optimizer_G.zero_grad()  
        with autocast():
            loss_G = self.backward_G()          # calculate gradients for all G's
        self.scalar.scale(loss_G).backward() 
        self.scalar.step(self.optimizer_G)
        self.scalar.update()
  
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F], True) #Multipath
        self.optimizer_D.zero_grad()  
        with autocast():
            loss_D = self.backward_D()      # calculate gradients for all D's
        self.scalar.scale(loss_D).backward()
        self.scalar.step(self.optimizer_D)
        self.scalar.update()