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


class ResnetMultipathCycleGANModel(BaseModel):
    """
    Utilizes 4 generators and 4 discriminators to perform the cycleGAN task.
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
            # parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_L2', type = float, default=10e6, help='Weight for L2 function between generated outout and input for a given path. Will begin decaying once the images are forced to identity.')
            parser.add_argument('--lambda_seg', type=float, default=0.01, help='weight for tissue statistic loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)


        # #For multipath cycleGAN: NEed to include the L2 losses 
        self.loss_names = ['D_AB', 'G_AB', 'L2SHSS','cycle_AB', 'segSHSS', 'D_BA', 'G_BA','L2SSSH','cycle_BA', 'segSSSH', #Siemens hard Siemens soft
                           'D_AC', 'G_AC', 'L2SHGH','cycle_AC', 'segSHGH', 'D_CA', 'G_CA','L2GHSH','cycle_CA', 'segGHSH', #Siemens hard GE hard
                           'D_AD', 'G_AD', 'L2SHGS','cycle_AD', 'segSHGS', 'D_DA', 'G_DA','L2GSSH','cycle_DA', 'segGSSH', #Siemens hard GE soft
                           'D_BC', 'G_BC', 'L2SSGH','cycle_BC', 'segSSGH', 'D_CB', 'G_CB','L2GHSS','cycle_CB', 'segGHSS', #Siemens soft GE hard
                           'D_BD', 'G_BD', 'L2SSGS','cycle_BD', 'segSSGS', 'D_DB', 'G_DB','L2GSSS','cycle_DB', 'segSSGS', #Siemens soft GE soft
                           'D_CD', 'G_CD', 'L2GHGS','cycle_CD', 'segGHGS', 'D_DC', 'G_DC','L2GSGH','cycle_DC', 'segGSGH'] #GE hard GE soft   
 
        # A = B50f, B = B30f, C = GE BONE, D = GE STD
        visual_names_AB = ['B50f', 'fake_BA', 'rec_AB'] #Siemens hard to soft
        visual_names_BA = ['B30f', 'fake_AB', 'rec_BA'] #siemens soft to hard
        visual_names_AC = ['B50f', 'fake_CA', 'rec_AC'] #Siemens hard to GE hard
        visual_names_CA = ['BONE', 'fake_AC', 'rec_CA'] #gE hard to Siemens hard
        visual_names_AD = ['B50f', 'fake_DA', 'rec_AD'] #Siemens hard to GE soft
        visual_names_DA = ['STD', 'fake_AD', 'rec_DA'] #GE soft to Siemens hard
        visual_names_BC = ['B30f', 'fake_CB', 'rec_BC'] #Siemens soft GE hard
        visual_names_CB = ['BONE', 'fake_BC', 'rec_CB'] #GE hahd siemens soft
        visual_names_BD = ['B30f', 'fake_DB', 'rec_BD'] #Siemens soft GE soft
        visual_names_DB = ['STD', 'fake_BD', 'rec_DB'] #GE soft Siemens soft
        visual_names_CD = ['BONE', 'fake_DC', 'rec_CD'] #GE hard siemens soft
        visual_names_DC = ['STD', 'fake_CD', 'rec_DC'] #GE soft siemens hard


        #Multipath GAN
        self.visual_names = visual_names_AB + visual_names_BA + visual_names_AC + visual_names_CA + visual_names_AD + visual_names_DA + visual_names_BC + visual_names_CB + visual_names_BD + visual_names_DB + visual_names_CD + visual_names_DC


        if self.isTrain:
            #Multipath cycleGAN
            self.model_names = ['G_SH_encoder', 'G_SS_decoder', 'G_SS_encoder', 'G_SH_decoder',
                                'G_GH_decoder', 'G_GH_encoder','G_GS_decoder', 'G_GS_encoder',
                                'D_A', 'D_B', 'D_C', 'D_D']
        else:  # during test time, only load Gs
            #Multipath GAN
            self.model_names = ['G_SH_encoder', 'G_SS_decoder', 'G_SS_encoder', 'G_SH_decoder',
                                'G_GH_decoder', 'G_GH_encoder','G_GS_decoder', 'G_GS_encoder']

        #initialize the latent space class here and pass it to the encoder so that all of them have the same latent embedding. Each encoder here is a seperate object, therefore it needs the same latent space.

        shared_latent = ResBlocklatent(n_blocks=9, ngf=64, norm_layer=nn.InstanceNorm2d, padding_type='reflect').to(self.device)
        #For multipath, we need to define the encoders and decoders seperately and then construct a generator. 
        #Multipath cycleGAN encoder-decoder initalizations
        self.netG_SH_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #Siemens hard encoder
        self.netG_SS_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #Siemens soft encoder
        self.netG_GH_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #GE hard encoder
        self.netG_GS_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #GE soft encoder

        self.netG_SH_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #Siemens hard decoder
        self.netG_SS_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #Siemens soft decoder
        self.netG_GH_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #GE hard decoder
        self.netG_GS_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #GE soft decoder


        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # Siemens hard
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # Siemens soft
            self.netD_C = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # GE hard
            self.netD_D = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) # GE soft


        if self.isTrain:
            #Multipath GAN:Need to create 12 fake image buffers
            self.fake_AB_pool = ImagePool(opt.pool_size) #Siemens soft to Siemens hard (B to A)
            self.fake_BA_pool = ImagePool(opt.pool_size) #Siemens hard to Siemens soft (A to B)
            self.fake_AC_pool = ImagePool(opt.pool_size) #GE hard to Siemens hard (C to A)
            self.fake_CA_pool = ImagePool(opt.pool_size) #Siemens hard to GE hard (A to C) 
            self.fake_AD_pool = ImagePool(opt.pool_size) #GE soft to Siemens hard (D to A)
            self.fake_DA_pool = ImagePool(opt.pool_size) #Siemens hard to GE soft (A to D)
            self.fake_BC_pool = ImagePool(opt.pool_size) #GE hard to Siemens soft (C to B)
            self.fake_CB_pool = ImagePool(opt.pool_size) #Siemens soft to GE hard (B to C)
            self.fake_BD_pool = ImagePool(opt.pool_size) #GE soft to siemens soft (D to B)
            self.fake_DB_pool = ImagePool(opt.pool_size) #Siemens soft to GE soft (B to D)
            self.fake_CD_pool = ImagePool(opt.pool_size) #GE soft to GE hard (D to C))
            self.fake_DC_pool = ImagePool(opt.pool_size) #GE hard to GE soft (C to D)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss() #cycle loss
            self.criterionIdt = torch.nn.L1Loss() #Identity loss
            self.L2 = torch.nn.MSELoss() # additional L2 loss between generated output and input for a given path
            self.criterionSeg = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_SH_encoder.parameters(),
                                                                self.netG_SS_decoder.parameters(),
                                                                self.netG_SS_encoder.parameters(),
                                                                self.netG_SH_decoder.parameters(),
                                                                self.netG_GH_decoder.parameters(),
                                                                self.netG_GH_encoder.parameters(),
                                                                self.netG_GS_decoder.parameters(),
                                                                self.netG_GS_encoder.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),
                                                                self.netD_B.parameters(),
                                                                self.netD_C.parameters(),
                                                                self.netD_D.parameters()),
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
        # Multipath GAN
        self.B50f = input['A'].to(self.device) #Real B50f
        self.B30f = input['B'].to(self.device) 
        self.BONE = input['C'].to(self.device)
        self.STD = input['D'].to(self.device)
        self.B50f_mask = input['A_mask'].to(self.device)
        self.B30f_mask = input['B_mask'].to(self.device)
        self.BONE_mask = input['C_mask'].to(self.device)
        self.STD_mask = input['D_mask'].to(self.device)
        self.B50fpath = input['A_paths']
        self.B30fpath = input['B_paths']
        self.BONEpath = input['C_paths']
        self.STDpath = input['D_paths']
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
            if torch.sum(label) > 5: #More than 5 pixels in the mask
                real_mean_A[i] = torch.mean(real_A[real_mask_A == label])
                fake_mean_B[i] = torch.mean(fake_B[real_mask_A == label])
            else: #Less than 5 pixels in the mask for a given label
                real_mean_A[i] = 0.0
                fake_mean_B[i] = 0.0
        
        for i, label in enumerate(unique_labels_backward):
            if torch.sum(label) > 5:
                real_mean_B[i] = torch.mean(real_B[real_mask_B == label])
                fake_mean_A[i] = torch.mean(fake_A[real_mask_B == label])
            else:
                real_mean_B[i] = 0.0
                fake_mean_A[i] = 0.0
        
        loss_seg_A = self.criterionSeg(real_mean_A, fake_mean_B) * self.opt.lambda_seg
        loss_seg_B = self.criterionSeg(real_mean_B, fake_mean_A) * self.opt.lambda_seg

        return loss_seg_A, loss_seg_B

    def cyclicpath(self, target_decoder, target_encoder, source_decoder, latent):
        """
        Code snippet for a given cyclic path in the multipath GAN model
        """
        fake_image = target_decoder(latent)
        latent_rec = target_encoder(fake_image)
        reconstructed = source_decoder(latent_rec)
        return fake_image, reconstructed
    
    def L2loss_decay(self, real_image, fake_image):
        interp_real = F.interpolate(real_image, size = [256,256], mode = 'bilinear', align_corners=True)
        interp_fake = F.interpolate(fake_image, size = [256,256], mode = 'bilinear', align_corners=True)
        L2loss = self.L2(interp_real, interp_fake)
        return L2loss

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with autocast():
            latentA = self.netG_SH_encoder(self.B50f) #encoding Real B50f
            latentB = self.netG_SS_encoder(self.B30f) #Encoding real B30f
            latentC = self.netG_GH_encoder(self.BONE) #Encoding real BONE
            latentD = self.netG_GS_encoder(self.STD) #encoding real STD

            #Generator 1: Siemens B50f -> B30f
            self.fake_BA, self.rec_AB = self.cyclicpath(self.netG_SS_decoder, self.netG_SS_encoder, self.netG_SH_decoder, latentA) #Siemens hard to soft
            #Generator 2: Siemens B30f -> B50f
            self.fake_AB, self.rec_BA = self.cyclicpath(self.netG_SH_decoder, self.netG_SH_encoder, self.netG_SS_decoder, latentB) #Siemens soft to hard
            #Generator 3: Siemens B50f -> GE BONE
            self.fake_CA, self.rec_AC = self.cyclicpath(self.netG_GH_decoder, self.netG_GH_encoder, self.netG_SH_decoder, latentA) #Siemens hard to GE hard
            #Generator 4: GE BONE -> Siemens B50f
            self.fake_AC, self.rec_CA = self.cyclicpath(self.netG_SH_decoder, self.netG_SH_encoder, self.netG_GH_decoder, latentC) #GE hard to Siemens hard
            #Generator 5: Siemens B50f -> GE STD
            self.fake_DA, self.rec_AD = self.cyclicpath(self.netG_GS_decoder, self.netG_GS_encoder, self.netG_SH_decoder, latentA) #Siemens hard to GE soft
            #Generator 6:GE STD -> Siemens B50f
            self.fake_AD, self.rec_DA = self.cyclicpath(self.netG_SH_decoder, self.netG_SH_encoder, self.netG_GS_decoder, latentD) #GE soft to Siemens hard
            #Generator 7: Siemens B30f -> GE BONE
            self.fake_CB, self.rec_BC = self.cyclicpath(self.netG_GH_decoder, self.netG_GH_encoder, self.netG_SS_decoder, latentB) #Siemens soft to GE hard
            #Genertaor 8:GE BONE -> Siemens B30f
            self.fake_BC, self.rec_CB = self.cyclicpath(self.netG_SS_decoder, self.netG_SS_encoder, self.netG_GH_decoder, latentC) #GE hard to Siemens soft
            #Generator 9: Siemens B30f -> GE STD
            self.fake_DB, self.rec_BD = self.cyclicpath(self.netG_GS_decoder, self.netG_GS_encoder, self.netG_SS_decoder, latentB) #Siemens soft to GE soft
            #Generator 10: GE STD -> Siemens B30f
            self.fake_BD, self.rec_DB = self.cyclicpath(self.netG_SS_decoder, self.netG_SS_encoder, self.netG_GS_decoder, latentD) #GE soft to Siemens soft
            #Generator 11: GE BONE -> GE STD
            self.fake_DC, self.rec_CD = self.cyclicpath(self.netG_GS_decoder, self.netG_GS_encoder, self.netG_GH_decoder, latentC) #GE hard to GE soft
            #Generator 12 : GE STD -> GE BONE
            self.fake_CD, self.rec_DC = self.cyclicpath(self.netG_GH_decoder, self.netG_GH_encoder, self.netG_GS_decoder, latentD) #GE soft to GE hard


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
        """Calculate GAN loss for discriminator D_AB"""
        fake_BA = self.fake_BA_pool.query(self.fake_BA)
        self.loss_D_AB = self.backward_D_basic(self.netD_A, self.B30f, fake_BA)

        """Calculate GAN loss for discriminator D_BA"""
        fake_AB = self.fake_AB_pool.query(self.fake_AB)
        self.loss_D_BA = self.backward_D_basic(self.netD_B, self.B50f, fake_AB)

        """Calculate GAN loss for discriminator D_AC"""
        fake_CA = self.fake_CA_pool.query(self.fake_CA)
        self.loss_D_AC = self.backward_D_basic(self.netD_A, self.BONE, fake_CA)

        """Calculate GAN loss for discriminator D_CA"""
        fake_AC = self.fake_AC_pool.query(self.fake_AC)
        self.loss_D_CA = self.backward_D_basic(self.netD_C, self.B50f, fake_AC)

        """Calculate GAN loss for discriminator D_AD"""
        fake_DA = self.fake_DA_pool.query(self.fake_DA)
        self.loss_D_AD = self.backward_D_basic(self.netD_A, self.STD, fake_DA)

        """Calculate GAN loss for discriminator D_DA"""
        fake_AD = self.fake_AD_pool.query(self.fake_AD)
        self.loss_D_DA = self.backward_D_basic(self.netD_D, self.B50f, fake_AD)

        """Calculate GAN loss for discriminator D_BC"""
        fake_CB = self.fake_CB_pool.query(self.fake_CB)
        self.loss_D_BC = self.backward_D_basic(self.netD_B, self.BONE, fake_CB)

        """Calculate GAN loss for discriminator D_CB"""
        fake_BC = self.fake_BC_pool.query(self.fake_BC)
        self.loss_D_CB = self.backward_D_basic(self.netD_C, self.B30f, fake_BC)

        """Calculate GAN loss for discriminator D_BD"""
        fake_DB = self.fake_DB_pool.query(self.fake_DB)
        self.loss_D_BD = self.backward_D_basic(self.netD_B, self.STD, fake_DB)

        """Calculate GAN loss for discriminator D_DB"""
        fake_BD = self.fake_BD_pool.query(self.fake_BD)
        self.loss_D_DB = self.backward_D_basic(self.netD_D, self.B30f, fake_BD)

        """Calculate GAN loss for discriminator D_CD"""
        fake_DC = self.fake_DC_pool.query(self.fake_DC)
        self.loss_D_CD = self.backward_D_basic(self.netD_C, self.STD, fake_DC)

        """Calculate GAN loss for discriminator D_CB"""
        fake_CD = self.fake_CD_pool.query(self.fake_CD)
        self.loss_D_DC = self.backward_D_basic(self.netD_D, self.BONE, fake_CD)

        self.loss_D = self.loss_D_AB + self.loss_D_BA + self.loss_D_AC + self.loss_D_CA + \
                      self.loss_D_AD + self.loss_D_DA + self.loss_D_BC + self.loss_D_CB + \
                      self.loss_D_BD + self.loss_D_DB + self.loss_D_CD + self.loss_D_DC

        return self.loss_D

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_L2 = self.opt.lambda_L2
        # GAN loss D_A(G_A(A))
        # GAN loss D_B(G_B(B))

        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)


        # Least squares loss for all the generators
        self.loss_G_AB = self.criterionGAN(self.netD_A(self.fake_BA), True)
        self.loss_G_BA = self.criterionGAN(self.netD_B(self.fake_AB), True)
        self.loss_G_AC = self.criterionGAN(self.netD_A(self.fake_CA), True)
        self.loss_G_CA = self.criterionGAN(self.netD_C(self.fake_AC), True)
        self.loss_G_AD = self.criterionGAN(self.netD_A(self.fake_DA), True)
        self.loss_G_DA = self.criterionGAN(self.netD_D(self.fake_AD), True)
        self.loss_G_BC = self.criterionGAN(self.netD_B(self.fake_CB), True)
        self.loss_G_CB = self.criterionGAN(self.netD_C(self.fake_BC), True)
        self.loss_G_BD = self.criterionGAN(self.netD_B(self.fake_DB), True)
        self.loss_G_DB = self.criterionGAN(self.netD_D(self.fake_BD), True)
        self.loss_G_CD = self.criterionGAN(self.netD_C(self.fake_DC), True)
        self.loss_G_DC = self.criterionGAN(self.netD_D(self.fake_CD), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # Backward cycle loss || G_A(G_B(B)) - B||

        #Vanilla CycleGAN
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        #Multipath cycleGAN: Cycle consistency losses
        self.loss_cycle_AB = self.criterionCycle(self.rec_AB, self.B50f) * lambda_A
        self.loss_cycle_BA = self.criterionCycle(self.rec_BA, self.B30f) * lambda_B
        self.loss_cycle_AC = self.criterionCycle(self.rec_AC, self.B50f) * lambda_A
        self.loss_cycle_CA = self.criterionCycle(self.rec_CA, self.BONE) * lambda_B
        self.loss_cycle_AD = self.criterionCycle(self.rec_AD, self.B50f) * lambda_A
        self.loss_cycle_DA = self.criterionCycle(self.rec_DA, self.STD) * lambda_B
        self.loss_cycle_BC = self.criterionCycle(self.rec_BC, self.B30f) * lambda_A
        self.loss_cycle_CB = self.criterionCycle(self.rec_CB, self.BONE) * lambda_B
        self.loss_cycle_BD = self.criterionCycle(self.rec_BD, self.B30f) * lambda_A
        self.loss_cycle_DB = self.criterionCycle(self.rec_DB, self.STD) * lambda_B
        self.loss_cycle_CD = self.criterionCycle(self.rec_CD, self.BONE) * lambda_A
        self.loss_cycle_DC = self.criterionCycle(self.rec_DC, self.STD) * lambda_B

        #Additional L2 loss for the objective function: Downsample real and fake tensors, compute MSE between them
        self.loss_L2SHSS = self.L2loss_decay(self.B50f, self.fake_BA) * lambda_L2
        self.loss_L2SSSH = self.L2loss_decay(self.B30f, self.fake_AB) * lambda_L2
        self.loss_L2SHGH = self.L2loss_decay(self.B50f, self.fake_CA) * lambda_L2
        self.loss_L2GHSH = self.L2loss_decay(self.BONE, self.fake_AC) * lambda_L2
        self.loss_L2SHGS = self.L2loss_decay(self.B50f, self.fake_DA) * lambda_L2
        self.loss_L2GSSH = self.L2loss_decay(self.STD, self.fake_AD) * lambda_L2
        self.loss_L2SSGH = self.L2loss_decay(self.B30f, self.fake_CB) * lambda_L2
        self.loss_L2GHSS = self.L2loss_decay(self.BONE, self.fake_BC) * lambda_L2
        self.loss_L2SSGS = self.L2loss_decay(self.B30f, self.fake_DB) * lambda_L2
        self.loss_L2GSSS = self.L2loss_decay(self.STD, self.fake_BD) * lambda_L2
        self.loss_L2GHGS = self.L2loss_decay(self.BONE, self.fake_DC) * lambda_L2
        self.loss_L2GSGH = self.L2loss_decay(self.STD, self.fake_CD) * lambda_L2
        
        #Tissue statistic loss
        self.loss_segSHSS, self.loss_segSSSH = self.tissue_statistic_loss(self.B50f, self.fake_AB, self.B30f, self.fake_BA, self.B50f_mask, self.B30f_mask)
        self.loss_segSHGH, self.loss_segGHSH = self.tissue_statistic_loss(self.B50f, self.fake_AC, self.BONE, self.fake_CA, self.B50f_mask, self.BONE_mask)
        self.loss_segSHGS, self.loss_segGSSH = self.tissue_statistic_loss(self.B50f, self.fake_AD, self.STD, self.fake_DA, self.B50f_mask, self.STD_mask)
        self.loss_segSSGH, self.loss_segGHSS = self.tissue_statistic_loss(self.B30f, self.fake_BC, self.BONE, self.fake_CB, self.B30f_mask, self.BONE_mask)
        self.loss_segSSGS, self.loss_segGSSS = self.tissue_statistic_loss(self.B30f, self.fake_BD, self.STD, self.fake_DB, self.B30f_mask, self.STD_mask)
        self.loss_segGHGS, self.loss_segGSGH = self.tissue_statistic_loss(self.BONE, self.fake_CD, self.STD, self.fake_DC, self.BONE_mask, self.STD_mask)


        #this is loss function for multipath cycleGAN: Adversarial losses + desicriminator losses + L2 losses
        self.loss_G = self.loss_G_AB + self.loss_G_BA + self.loss_cycle_AB + self.loss_cycle_BA + \
                       self.loss_G_AC + self.loss_G_CA + self.loss_cycle_AC + self.loss_cycle_CA + \
                       self.loss_G_AD + self.loss_G_DA + self.loss_cycle_AD + self.loss_cycle_DA + \
                       self.loss_G_BC + self.loss_G_CB + self.loss_cycle_BC + self.loss_cycle_CB + \
                       self.loss_G_BD + self.loss_G_DB + self.loss_cycle_BD + self.loss_cycle_DB + \
                       self.loss_G_CD + self.loss_G_DC + self.loss_cycle_CD + self.loss_cycle_DC +\
                       self.loss_L2SHSS + self.loss_L2SSSH + self.loss_L2SHGH + self.loss_L2GHSH + self.loss_L2SHGS + self.loss_L2GSSH + \
                       self.loss_L2SSGH + self.loss_L2GHSS + self.loss_L2SSGS + self.loss_L2GSSS + self.loss_L2GHGS + self.loss_L2GSGH + \
                       self.loss_segSHSS + self.loss_segSSSH + self.loss_segSHGH + self.loss_segGHSH + self.loss_segSHGS + self.loss_segGSSH + \
                       self.loss_segSSGH + self.loss_segGHSS + self.loss_segSSGS + self.loss_segGSSS + self.loss_segGHGS + self.loss_segGSGH

        return self.loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D], False) #Multipath 
        self.optimizer_G.zero_grad()  # set G gradients to zero
        with autocast():
            loss_G = self.backward_G()          # calculate gradients for all G's
        self.scalar.scale(loss_G).backward() 
        self.scalar.step(self.optimizer_G)
        self.scalar.update()
     # update weights for the encoders and decoders
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D], True) #Multipath
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        with autocast():
            loss_D = self.backward_D()      # calculate gradients for D_A
        self.scalar.scale(loss_D).backward()
        self.scalar.step(self.optimizer_D)
        self.scalar.update()