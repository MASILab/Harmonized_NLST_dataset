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


class ResnetMultipathWithoutIdentityStageTwoCycleGANModel(BaseModel):
    """
    Train stage two multipath cycleGAN by freezing models for Philips B, C and D kernels. Freeze encoder, decoder models for B50f, B30f, BONE, LUNG, STANDARD and B80f.
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
        # Older kernels: A - Siemens B50f, B - Siemens B30f, C - GE BONE, D - GE STD, E - GE LUNG, F - Siemens B80f 
        # Newer kernels: G - Philips B, H - Philips C, I - Philips D
        self.loss_names = ['D_AG', 'G_AG','cycle_AG', 'segB50fPhilB', 'D_GA', 'G_GA','cycle_GA', 'segPhilBB50f', 
                           'D_AH', 'G_AH','cycle_AH', 'segB50fPhilC', 'D_HA', 'G_HA','cycle_HA', 'segPhilCB50f', 
                           'D_AI', 'G_AI','cycle_AI', 'segB50fPhilD', 'D_IA', 'G_IA','cycle_IA', 'segPhilDB50f', 
                           'D_BG', 'G_BG','cycle_BG', 'segB30fPhilB', 'D_GB', 'G_GB','cycle_GB', 'segPhilBB30f',
                           'D_BH', 'G_BH','cycle_BH', 'segB30fPhilC', 'D_HB', 'G_HB','cycle_HB', 'segPhilCB30f',
                           'D_BI', 'G_BI','cycle_BI', 'segB30fPhilD', 'D_IB', 'G_IB','cycle_IB', 'segPhilDB30f',
                           'D_CG', 'G_CG','cycle_CG', 'segBONEPhilB', 'D_GC', 'G_GC','cycle_GC', 'segPhilBBONE',
                           'D_CH', 'G_CH','cycle_CH', 'segBONEPhilC', 'D_HC', 'G_HC','cycle_HC', 'segPhilCBONE',
                           'D_CI', 'G_CI','cycle_CI', 'segBONEPhilD', 'D_IC', 'G_IC','cycle_IC', 'segPhilDBONE',
                           'D_DG', 'G_DG','cycle_DG', 'segSTDPhilB', 'D_GD', 'G_GD','cycle_GD', 'segPhilBSTD',
                           'D_DH', 'G_DH','cycle_DH', 'segSTDPhilC', 'D_HD', 'G_HD','cycle_HD', 'segPhilCSTD',
                           'D_DI', 'G_DI','cycle_DI', 'segSTDPhilD', 'D_ID', 'G_ID','cycle_ID', 'segPhilDSTD',
                           'D_EG', 'G_EG','cycle_EG', 'segLUNGPhilB', 'D_GE', 'G_GE','cycle_GE', 'segPhilBLUNG',
                           'D_EH', 'G_EH','cycle_EH', 'segLUNGPhilC', 'D_HE', 'G_HE','cycle_HE', 'segPhilCLUNG',
                           'D_EI', 'G_EI','cycle_EI', 'segLUNGPhilD', 'D_IE', 'G_IE','cycle_IE', 'segPhilDLUNG',
                           'D_FG', 'G_FG','cycle_FG', 'segB80fPhilB', 'D_GF', 'G_GF','cycle_GF', 'segPhilBB80f',
                           'D_FH', 'G_FH','cycle_FH', 'segB80fPhilC', 'D_HF', 'G_HF','cycle_HF', 'segPhilCB80f',
                           'D_FI', 'G_FI','cycle_FI', 'segB80fPhilD', 'D_IF', 'G_IF','cycle_IF', 'segPhilDB80f',
                           'D_GH', 'G_GH','cycle_GH', 'segPhilBPhilC', 'D_HG', 'G_HG','cycle_HG', 'segPhilCPhilB',
                           'D_GI', 'G_GI','cycle_GI', 'segPhilBPhilD', 'D_IG', 'G_IG','cycle_IG', 'segPhilDPhilB',
                           'D_HI', 'G_HI','cycle_HI', 'segPhilCPhilD', 'D_IH', 'G_IH','cycle_IH', 'segPhilDPhilC']
 
        # A = B50f, B = B30f, C = GE BONE, D = GE STD, E = GE LUNG, F = Siemens B80f, G = Philips B, H = Philips C, I = Philips D
        visual_names_AG = ['B50f', 'fake_GA', 'rec_AG'] 
        visual_names_GA = ['B', 'fake_AG', 'rec_GA'] 

        visual_names_AH = ['B50f', 'fake_HA', 'rec_AH']
        visual_names_HA = ['C', 'fake_AH', 'rec_HA']

        visual_names_AI = ['B50f', 'fake_IA', 'rec_AI']
        visual_names_IA = ['D', 'fake_AI', 'rec_IA']

        visual_names_BG = ['B30f', 'fake_GB', 'rec_BG']
        visual_names_GB = ['B', 'fake_BG', 'rec_GB']

        visual_names_BH = ['B30f', 'fake_HB', 'rec_BH']
        visual_names_HB = ['C', 'fake_BH', 'rec_HB']

        visual_names_BI = ['B30f', 'fake_IB', 'rec_BI']
        visual_names_IB = ['D', 'fake_BI', 'rec_IB']

        visual_names_CG = ['BONE', 'fake_GC', 'rec_CG']
        visual_names_GC = ['B', 'fake_CG', 'rec_GC']

        visual_names_CH = ['BONE', 'fake_HC', 'rec_CH']
        visual_names_HC = ['C', 'fake_CH', 'rec_HC']

        visual_names_CI = ['BONE', 'fake_IC', 'rec_CI']
        visual_names_IC = ['D', 'fake_CI', 'rec_IC']

        visual_names_DG = ['STD', 'fake_GD', 'rec_DG']
        visual_names_GD = ['B', 'fake_DG', 'rec_GD']

        visual_names_DH = ['STD', 'fake_HD', 'rec_DH']
        visual_names_HD = ['C', 'fake_DH', 'rec_HD']

        visual_names_DI = ['STD', 'fake_ID', 'rec_DI']
        visual_names_ID = ['D', 'fake_DI', 'rec_ID']

        visual_names_EG = ['LUNG', 'fake_GE', 'rec_EG']
        visual_names_GE = ['B', 'fake_EG', 'rec_GE']

        visual_names_EH = ['LUNG', 'fake_HE', 'rec_EH']
        visual_names_HE = ['C', 'fake_EH', 'rec_HE']

        visual_names_EI = ['LUNG', 'fake_IE', 'rec_EI']
        visual_names_IE = ['D', 'fake_EI', 'rec_IE']

        visual_names_FG = ['B80f', 'fake_GF', 'rec_FG']
        visual_names_GF = ['B', 'fake_FG', 'rec_GF']

        visual_names_FH = ['B80f', 'fake_HF', 'rec_FH']
        visual_names_HF = ['C', 'fake_FH', 'rec_HF']

        visual_names_FI = ['B80f', 'fake_IF', 'rec_FI']
        visual_names_IF = ['D', 'fake_FI', 'rec_IF']

        visual_names_GH = ['B', 'fake_HG', 'rec_GH']
        visual_names_HG = ['C', 'fake_GH', 'rec_HG']

        visual_names_GI = ['B', 'fake_IG', 'rec_GI']
        visual_names_IG = ['D', 'fake_GI', 'rec_IG']

        visual_names_HI = ['C', 'fake_IH', 'rec_HI']
        visual_names_IH = ['D', 'fake_HI', 'rec_IH']


        #Multipath GAN
        self.visual_names = visual_names_AG + visual_names_GA + visual_names_AH + visual_names_HA + visual_names_AI + visual_names_IA + \
                            visual_names_BG + visual_names_GB + visual_names_BH + visual_names_HB + visual_names_BI + visual_names_IB + \
                            visual_names_CG + visual_names_GC + visual_names_CH + visual_names_HC + visual_names_CI + visual_names_IC + \
                            visual_names_DG + visual_names_GD + visual_names_DH + visual_names_HD + visual_names_DI + visual_names_ID + \
                            visual_names_EG + visual_names_GE + visual_names_EH + visual_names_HE + visual_names_EI + visual_names_IE + \
                            visual_names_FG + visual_names_GF + visual_names_FH + visual_names_HF + visual_names_FI + visual_names_IF + \
                            visual_names_GH + visual_names_HG + visual_names_GI + visual_names_IG + visual_names_HI + visual_names_IH


        if self.isTrain:
            self.model_names = ['G_B50f_encoder', 'G_B50f_decoder', 'G_B30f_encoder', 'G_B30f_decoder',
                                'G_BONE_encoder', 'G_BONE_decoder','G_STD_encoder', 'G_STD_decoder',
                                'G_LUNG_encoder', 'G_LUNG_decoder','G_B80f_encoder', 'G_B80f_decoder',
                                'G_B_encoder', 'G_B_decoder', 'G_C_encoder', 'G_C_decoder', 'G_D_encoder', 'G_D_decoder',
                                'D_A', 'D_B', 'D_C', 'D_D', 'D_E', 'D_F', 'D_G', 'D_H', 'D_I']
            
        else:  # during test time, only load Gs
            self.model_names = ['G_B50f_encoder', 'G_B50f_decoder', 'G_B30f_encoder', 'G_B30f_decoder',
                                'G_BONE_decoder', 'G_BONE_encoder','G_STD_decoder', 'G_STD_encoder',
                                'G_LUNG_decoder', 'G_LUNG_encoder','G_B80f_decoder', 'G_B80f_encoder', 
                                'G_B_encoder', 'G_B_decoder', 'G_C_encoder', 'G_C_decoder', 'G_D_encoder', 'G_D_decoder']


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

        self.netG_B_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #B encoder
        self.netG_B_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #B decoder

        self.netG_C_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #C encoder
        self.netG_C_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #C decoder

        self.netG_D_encoder = networks.G_encoder(opt.input_nc, opt.ngf, opt.netG_encoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, latent_layer = shared_latent) #D encoder
        self.netG_D_decoder = networks.G_decoder(opt.output_nc, opt.ngf, opt.netG_decoder, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids) #D decoder

        
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
            self.netD_G = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #Philips B 
            self.netD_H = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #Philips C
            self.netD_I = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids) #Philips D


        if self.isTrain:

            self.fake_pools = {}
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            excluded_combinations = ['AB', 'BA', 'AC', 'CA', 'AD', 'DA', 'AE', 'EA', 'AF', 'FA', 'BC', 'CB', 'BD', 'DB', 'BE', 'EB', 'BF', 'FB', 
                                     'CD', 'DC', 'CE', 'EC', 'CF', 'FC', 'DE', 'ED', 'DF', 'FD', 'EF', 'FE', 'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II']

            for combination in itertools.product(letters, repeat=2):
                key = ''.join(combination)
                if key not in excluded_combinations:
                    self.fake_pools['fake_' + key + '_pool'] = ImagePool(opt.pool_size)
            
            print("---------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------")
            print(f"Created image buffers for all fake images. Length of dictionary with image buffers is {len(self.fake_pools)}")
            print("---------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------")


            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss() #cycle loss
            self.criterionSeg = torch.nn.MSELoss() 

            self.set_requires_grad([self.netG_B50f_encoder, self.netG_B50f_decoder, self.netG_B30f_encoder, self.netG_B30f_decoder, 
                                   self.netG_BONE_encoder, self.netG_BONE_decoder, self.netG_STD_encoder, self.netG_STD_decoder,
                                   self.netG_LUNG_encoder, self.netG_LUNG_decoder, self.netG_B80f_encoder, self.netG_B80f_decoder], False)
            
            print("---------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------")
            print("Freezing the following models: ")
            print("Siemens B50f encoder and decoder, Siemens B30f encoder and decoder, GE BONE encoder and decoder, GE STD encoder and decoder, GE LUNG encoder and decoder, Siemens B80f encoder and decoder")
            print("---------------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------------")

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_B_encoder.parameters(),
                                                                self.netG_B_decoder.parameters(),
                                                                self.netG_C_encoder.parameters(),
                                                                self.netG_C_decoder.parameters(),
                                                                self.netG_D_encoder.parameters(),
                                                                self.netG_D_decoder.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(),
                                                                self.netD_B.parameters(),
                                                                self.netD_C.parameters(),
                                                                self.netD_D.parameters(),
                                                                self.netD_E.parameters(),
                                                                self.netD_F.parameters(),
                                                                self.netD_G.parameters(),
                                                                self.netD_H.parameters(),
                                                                self.netD_I.parameters()),
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
        self.B50f = input['B50f_image'].to(self.device) #A
        self.B30f = input['B30f_image'].to(self.device) #B
        self.BONE = input['BONE_image'].to(self.device) #C
        self.STD = input['STANDARD_image'].to(self.device) #D
        self.B80f = input['B80f_image'].to(self.device) #E
        self.LUNG = input['LUNG_image'].to(self.device) #F
        self.B = input['B_image'].to(self.device) #G
        self.C = input['C_image'].to(self.device) #H
        self.D = input['D_image'].to(self.device) #I

        self.B50f_mask = input['B50f_mask'].to(self.device)
        self.B30f_mask = input['B30f_mask'].to(self.device)
        self.BONE_mask = input['BONE_mask'].to(self.device)
        self.STD_mask = input['STANDARD_mask'].to(self.device)
        self.LUNG_mask = input['LUNG_mask'].to(self.device)
        self.B80f_mask = input['B80f_mask'].to(self.device)
        self.B_mask = input['B_mask'].to(self.device)
        self.C_mask = input['C_mask'].to(self.device)
        self.D_mask = input['D_mask'].to(self.device)

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
            if torch.sum(real_mask_A == label) > 5: 
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
            latent_B = self.netG_B_encoder(self.B) #G
            latent_C = self.netG_C_encoder(self.C) #H
            latent_D = self.netG_D_encoder(self.D) #I

            #Siemens B50f
            #Generator 1: B50f to Philips B
            self.fake_GA, self.rec_AG = self.cyclicpath(latent_B50f, self.netG_B_decoder, self.netG_B_encoder, self.netG_B50f_decoder) #B50f to B
            #Generator 2: Philips B to B50f
            self.fake_AG, self.rec_GA = self.cyclicpath(latent_B, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_B_decoder) #B to B50f
            #Generator 3: B50f to Philips C
            self.fake_HA, self.rec_AH = self.cyclicpath(latent_B50f, self.netG_C_decoder, self.netG_C_encoder, self.netG_B50f_decoder) #B50f to C
            #Generator 4: Philips C to B50f
            self.fake_AH, self.rec_HA = self.cyclicpath(latent_C, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_C_decoder) #C to B50f
            #Generator 5: B50f to Philips D
            self.fake_IA, self.rec_AI = self.cyclicpath(latent_B50f, self.netG_D_decoder, self.netG_D_encoder, self.netG_B50f_decoder) #B50f to D
            #Generator 6: Philips D to B50f
            self.fake_AI, self.rec_IA = self.cyclicpath(latent_D, self.netG_B50f_decoder, self.netG_B50f_encoder, self.netG_D_decoder) #D to B50f 

            #Siemens B30f
            #Generator 7: B30f to Philips B
            self.fake_GB, self.rec_BG = self.cyclicpath(latent_B30f, self.netG_B_decoder, self.netG_B_encoder, self.netG_B30f_decoder) #B30f to B
            #Generator 8: Philips B to B30f
            self.fake_BG, self.rec_GB = self.cyclicpath(latent_B, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_B_decoder) #B to B30f
            #Generator 9: B30f to Philips C
            self.fake_HB, self.rec_BH = self.cyclicpath(latent_B30f, self.netG_C_decoder, self.netG_C_encoder, self.netG_B30f_decoder) #B30f to C
            #Generator 10: Philips C to B30f
            self.fake_BH, self.rec_HB = self.cyclicpath(latent_C, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_C_decoder) #C to B30f
            #Generator 11: B30f to Philips D
            self.fake_IB, self.rec_BI = self.cyclicpath(latent_B30f, self.netG_D_decoder, self.netG_D_encoder, self.netG_B30f_decoder) #B30f to D
            #Generator 12: Philips D to B30f
            self.fake_BI, self.rec_IB = self.cyclicpath(latent_D, self.netG_B30f_decoder, self.netG_B30f_encoder, self.netG_D_decoder) #D to B30f

            #GE BONE
            #Generator 13: BONE to Philips B
            self.fake_GC, self.rec_CG = self.cyclicpath(latent_BONE, self.netG_B_decoder, self.netG_B_encoder, self.netG_BONE_decoder) #BONE to B
            #Generator 14: Philips B to BONE
            self.fake_CG, self.rec_GC = self.cyclicpath(latent_B, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_B_decoder) #B to BONE
            #Generator 15: BONE to Philips C
            self.fake_HC, self.rec_CH = self.cyclicpath(latent_BONE, self.netG_C_decoder, self.netG_C_encoder, self.netG_BONE_decoder) #BONE to C
            #Generator 16: Philips C to BONE
            self.fake_CH, self.rec_HC = self.cyclicpath(latent_C, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_C_decoder) #C to BONE
            #Generator 17: BONE to Philips D
            self.fake_IC, self.rec_CI = self.cyclicpath(latent_BONE, self.netG_D_decoder, self.netG_D_encoder, self.netG_BONE_decoder) #BONE to D
            #Generator 18: Philips D to BONE
            self.fake_CI, self.rec_IC = self.cyclicpath(latent_D, self.netG_BONE_decoder, self.netG_BONE_encoder, self.netG_D_decoder) #D to BONE

            #GE STD
            #Generator 19: STD to Philips B
            self.fake_GD, self.rec_DG = self.cyclicpath(latent_STD, self.netG_B_decoder, self.netG_B_encoder, self.netG_STD_decoder) #STD
            #Generator 20: Philips B to STD
            self.fake_DG, self.rec_GD = self.cyclicpath(latent_B, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_B_decoder) #B
            #Generator 21: STD to Philips C
            self.fake_HD, self.rec_DH = self.cyclicpath(latent_STD, self.netG_C_decoder, self.netG_C_encoder, self.netG_STD_decoder) #STD to C
            #Generator 22: Philips C to STD
            self.fake_DH, self.rec_HD = self.cyclicpath(latent_C, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_C_decoder) #C to STD
            #Generator 23: STD to Philips D
            self.fake_ID, self.rec_DI = self.cyclicpath(latent_STD, self.netG_D_decoder, self.netG_D_encoder, self.netG_STD_decoder) #STD to D
            #Generator 24: Philips D to STD
            self.fake_DI, self.rec_ID = self.cyclicpath(latent_D, self.netG_STD_decoder, self.netG_STD_encoder, self.netG_D_decoder) #D to STD

            #GE LUNG
            #Generator 25: LUNG to Philips B
            self.fake_GE, self.rec_EG = self.cyclicpath(latent_LUNG, self.netG_B_decoder, self.netG_B_encoder, self.netG_LUNG_decoder) #LUNG to B
            #Generator 26: Philips B to LUNG
            self.fake_EG, self.rec_GE = self.cyclicpath(latent_B, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_B_decoder) #B to LUNG
            #Generator 27: LUNG to Philips C
            self.fake_HE, self.rec_EH = self.cyclicpath(latent_LUNG, self.netG_C_decoder, self.netG_C_encoder, self.netG_LUNG_decoder) #LUNG to C
            #Generator 28: Philips C to LUNG
            self.fake_EH, self.rec_HE = self.cyclicpath(latent_C, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_C_decoder) #C to LUNG
            #Generator 29: LUNG to Philips D
            self.fake_IE, self.rec_EI = self.cyclicpath(latent_LUNG, self.netG_D_decoder, self.netG_D_encoder, self.netG_LUNG_decoder) #LUNG to D
            #Generator 30: Philips D to LUNG
            self.fake_EI, self.rec_IE = self.cyclicpath(latent_D, self.netG_LUNG_decoder, self.netG_LUNG_encoder, self.netG_D_decoder) #D to LUNG

            #Siemens B80f
            #Generator 31: B80f to Philips B
            self.fake_GF, self.rec_FG = self.cyclicpath(latent_B80f, self.netG_B_decoder, self.netG_B_encoder, self.netG_B80f_decoder) #B80f to B
            #Generator 32: Philips B to B80f
            self.fake_FG, self.rec_GF = self.cyclicpath(latent_B, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_B_decoder) #B to B80f
            #Generator 33: B80f to Philips C    
            self.fake_HF, self.rec_FH = self.cyclicpath(latent_B80f, self.netG_C_decoder, self.netG_C_encoder, self.netG_B80f_decoder) #B80f to C
            #Generator 34: Philips C to B80f
            self.fake_FH, self.rec_HF = self.cyclicpath(latent_C, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_C_decoder) #C to B80f
            #Generator 35: B80f to Philips D
            self.fake_IF, self.rec_FI = self.cyclicpath(latent_B80f, self.netG_D_decoder, self.netG_D_encoder, self.netG_B80f_decoder) #B80f to D
            #Generator 36: Philips D to B80f
            self.fake_FI, self.rec_IF = self.cyclicpath(latent_D, self.netG_B80f_decoder, self.netG_B80f_encoder, self.netG_D_decoder) #D to B80f

            #Philips B
            #Generator 37: Philips B to Philips C
            self.fake_HG, self.rec_GH = self.cyclicpath(latent_B, self.netG_C_decoder, self.netG_C_encoder, self.netG_B_decoder) #B to C
            #Generator 38: Philips C to Philips B
            self.fake_GH, self.rec_HG = self.cyclicpath(latent_C, self.netG_B_decoder, self.netG_B_encoder, self.netG_C_decoder) #C to B
            #Generator 39: Philips B to Philips D
            self.fake_IG, self.rec_GI = self.cyclicpath(latent_B, self.netG_D_decoder, self.netG_D_encoder, self.netG_B_decoder) #B to D
            #Generator 40: Philips D to Philips B
            self.fake_GI, self.rec_IG = self.cyclicpath(latent_D, self.netG_B_decoder, self.netG_B_encoder, self.netG_D_decoder) #D to B

            #Philips C
            #Generator 41: Philips C to Philips D
            self.fake_IH, self.rec_HI = self.cyclicpath(latent_C, self.netG_D_decoder, self.netG_D_encoder, self.netG_C_decoder) #C to D
            #Generator 42: Philips D to Philips C
            self.fake_HI, self.rec_IH = self.cyclicpath(latent_D, self.netG_C_decoder, self.netG_C_encoder, self.netG_D_decoder) #D to C


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
    
    # Multipath cycleGAN
    def backward_D(self):
        #Siemens B50f (A)
        """Calculate GAN loss for discriminator D_AG (B50f to B)"""
        fake_GA = self.fake_pools["fake_GA_pool"].query(self.fake_GA)
        self.loss_D_AG = self.backward_D_basic(self.netD_A, self.B, fake_GA)

        """Calculate GAN loss for discriminator D_GA (B to B50f)"""
        fake_AG = self.fake_pools["fake_AG_pool"].query(self.fake_AG)
        self.loss_D_GA = self.backward_D_basic(self.netD_G, self.B50f, fake_AG) 

        """Calculate GAN loss for discriminator D_AH (B50f to C)"""
        fake_HA = self.fake_pools["fake_HA_pool"].query(self.fake_HA)
        self.loss_D_AH = self.backward_D_basic(self.netD_A, self.C, fake_HA)

        """Calculate GAN loss for discriminator D_HA (C to B50f)"""
        fake_AH = self.fake_pools["fake_AH_pool"].query(self.fake_AH)
        self.loss_D_HA = self.backward_D_basic(self.netD_H, self.B50f, fake_AH)

        """Calculate GAN loss for discriminator D_AI (B50f to D)"""
        fake_IA = self.fake_pools["fake_IA_pool"].query(self.fake_IA)
        self.loss_D_AI = self.backward_D_basic(self.netD_A, self.D, fake_IA) 

        """Calculate GAN loss for discriminator D_IA (D to B50f)"""
        fake_AI = self.fake_pools["fake_AI_pool"].query(self.fake_AI)
        self.loss_D_IA = self.backward_D_basic(self.netD_I, self.B50f, fake_AI)


        #Siemens B30f (B)
        """Calculate GAN loss for discriminator D_BG (B30f to B)"""
        fake_GB = self.fake_pools["fake_GB_pool"].query(self.fake_GB)
        self.loss_D_BG = self.backward_D_basic(self.netD_B, self.B, fake_GB)

        """Calculate GAN loss for discriminator D_GB (B to B30f)"""
        fake_BG = self.fake_pools["fake_BG_pool"].query(self.fake_BG)
        self.loss_D_GB = self.backward_D_basic(self.netD_G, self.B30f, fake_BG)

        """Calculate GAN loss for discriminator D_BH (B30f to C)"""
        fake_HB = self.fake_pools["fake_HB_pool"].query(self.fake_HB)
        self.loss_D_BH = self.backward_D_basic(self.netD_B, self.C, fake_HB)

        """Calculate GAN loss for discriminator D_HB (C to B30f)"""
        fake_BH = self.fake_pools["fake_BH_pool"].query(self.fake_BH)
        self.loss_D_HB = self.backward_D_basic(self.netD_H, self.B30f, fake_BH)

        """Calculate GAN loss for discriminator D_BI (B30f to D)"""
        fake_IB = self.fake_pools["fake_IB_pool"].query(self.fake_IB)
        self.loss_D_BI = self.backward_D_basic(self.netD_B, self.D, fake_IB)

        """Calculate GAN loss for discriminator D_IB (D to B30f)"""
        fake_BI = self.fake_pools["fake_BI_pool"].query(self.fake_BI)
        self.loss_D_IB = self.backward_D_basic(self.netD_I, self.B30f, fake_BI)


        #GE BONE (C)
        """Calculate GAN loss for discriminator D_CG (BONE to B)"""
        fake_GC = self.fake_pools["fake_GC_pool"].query(self.fake_GC)
        self.loss_D_CG = self.backward_D_basic(self.netD_C, self.B, fake_GC)

        """Calculate GAN loss for discriminator D_GC (B to BONE)"""
        fake_CG = self.fake_pools["fake_CG_pool"].query(self.fake_CG)
        self.loss_D_GC = self.backward_D_basic(self.netD_G, self.BONE, fake_CG)

        """Calculate GAN loss for discriminator D_CH (BONE to C)"""
        fake_HC = self.fake_pools["fake_HC_pool"].query(self.fake_HC)
        self.loss_D_CH = self.backward_D_basic(self.netD_C, self.C, fake_HC)

        """Calculate GAN loss for discriminator D_HC (C to BONE)"""
        fake_CH = self.fake_pools["fake_CH_pool"].query(self.fake_CH)
        self.loss_D_HC = self.backward_D_basic(self.netD_H, self.BONE, fake_CH)

        """Calculate GAN loss for discriminator D_CI (BONE to D)"""
        fake_IC = self.fake_pools["fake_IC_pool"].query(self.fake_IC)
        self.loss_D_CI = self.backward_D_basic(self.netD_C, self.D, fake_IC)

        """Calculate GAN loss for discriminator D_IC (D to BONE)"""
        fake_CI = self.fake_pools["fake_CI_pool"].query(self.fake_CI)
        self.loss_D_IC = self.backward_D_basic(self.netD_I, self.BONE, fake_CI)


        #GE STANDARD (D)
        """Calculate GAN loss for discriminator D_DG (STD to B)"""
        fake_GD = self.fake_pools["fake_GD_pool"].query(self.fake_GD)
        self.loss_D_DG = self.backward_D_basic(self.netD_D, self.B, fake_GD)

        """Calculate GAN loss for discriminator D_GD (B to STD)"""
        fake_DG = self.fake_pools["fake_DG_pool"].query(self.fake_DG)
        self.loss_D_GD = self.backward_D_basic(self.netD_G, self.STD, fake_DG)

        """Calculate GAN loss for discriminator D_DH (STD to C)"""
        fake_HD = self.fake_pools["fake_HD_pool"].query(self.fake_HD)
        self.loss_D_DH = self.backward_D_basic(self.netD_D, self.C, fake_HD)

        """Calculate GAN loss for discriminator D_HD (C to STD)"""
        fake_DH = self.fake_pools["fake_DH_pool"].query(self.fake_DH)
        self.loss_D_HD = self.backward_D_basic(self.netD_H, self.STD, fake_DH)

        """Calculate GAN loss for discriminator D_DI (STD to D)"""
        fake_ID = self.fake_pools["fake_ID_pool"].query(self.fake_ID)
        self.loss_D_DI = self.backward_D_basic(self.netD_D, self.D, fake_ID)

        """Calculate GAN loss for discriminator D_ID (D to STD)"""
        fake_DI = self.fake_pools["fake_DI_pool"].query(self.fake_DI)
        self.loss_D_ID = self.backward_D_basic(self.netD_I, self.STD, fake_DI)


        #GE LUNG (E)
        """Calculate GAN loss for discriminator D_EG (LUNG to B)"""
        fake_GE = self.fake_pools["fake_GE_pool"].query(self.fake_GE)
        self.loss_D_EG = self.backward_D_basic(self.netD_E, self.B, fake_GE)

        """Calculate GAN loss for discriminator D_GE (B to LUNG)"""
        fake_EG = self.fake_pools["fake_EG_pool"].query(self.fake_EG)
        self.loss_D_GE = self.backward_D_basic(self.netD_G, self.LUNG, fake_EG)

        """Calculate GAN loss for discriminator D_EH (LUNG to C)"""
        fake_HE = self.fake_pools["fake_HE_pool"].query(self.fake_HE)
        self.loss_D_EH = self.backward_D_basic(self.netD_E, self.C, fake_HE)

        """Calculate GAN loss for discriminator D_HE (C to LUNG)"""
        fake_EH = self.fake_pools["fake_EH_pool"].query(self.fake_EH)
        self.loss_D_HE = self.backward_D_basic(self.netD_H, self.LUNG, fake_EH)

        """Calculate GAN loss for discriminator D_EI (LUNG to D)"""
        fake_IE = self.fake_pools["fake_IE_pool"].query(self.fake_IE)
        self.loss_D_EI = self.backward_D_basic(self.netD_E, self.D, fake_IE)

        """Calculate GAN loss for discriminator D_IE (D to LUNG)"""
        fake_EI = self.fake_pools["fake_EI_pool"].query(self.fake_EI)
        self.loss_D_IE = self.backward_D_basic(self.netD_I, self.LUNG, fake_EI)


        #Siemns B80f (F)
        """Calculate GAN loss for discriminator D_FG (B80f to B)"""
        fake_GF = self.fake_pools["fake_GF_pool"].query(self.fake_GF)
        self.loss_D_FG = self.backward_D_basic(self.netD_F, self.B, fake_GF)

        """Calculate GAN loss for discriminator D_GF (B to B80f)"""
        fake_FG = self.fake_pools["fake_FG_pool"].query(self.fake_FG)
        self.loss_D_GF = self.backward_D_basic(self.netD_G, self.B80f, fake_FG)

        """Calculate GAN loss for discriminator D_FH (B80f to C)"""
        fake_HF = self.fake_pools["fake_HF_pool"].query(self.fake_HF)
        self.loss_D_FH = self.backward_D_basic(self.netD_F, self.C, fake_HF)

        """Calculate GAN loss for discriminator D_HF (C to B80f)"""
        fake_FH = self.fake_pools["fake_FH_pool"].query(self.fake_FH)
        self.loss_D_HF = self.backward_D_basic(self.netD_H, self.B80f, fake_FH) 

        """Calculate GAN loss for discriminator D_FI (B80f to D)"""
        fake_IF = self.fake_pools["fake_IF_pool"].query(self.fake_IF)
        self.loss_D_FI = self.backward_D_basic(self.netD_F, self.D, fake_IF)

        """Calculate GAN loss for discriminator D_IF (D to B80f)"""
        fake_FI = self.fake_pools["fake_FI_pool"].query(self.fake_FI)
        self.loss_D_IF = self.backward_D_basic(self.netD_I, self.B80f, fake_FI)


        #Philips B (G)
        """Calculate GAN loss for discriminator D_GH (B to C)"""
        fake_HG = self.fake_pools["fake_HG_pool"].query(self.fake_HG)
        self.loss_D_GH = self.backward_D_basic(self.netD_G, self.C, fake_HG)

        """Calculate GAN loss for discriminator D_HG (C to B)"""
        fake_GH = self.fake_pools["fake_GH_pool"].query(self.fake_GH)
        self.loss_D_HG = self.backward_D_basic(self.netD_H, self.B, fake_GH)

        """Calculate GAN loss for discriminator D_GI (B to D)"""
        fake_IG = self.fake_pools["fake_IG_pool"].query(self.fake_IG)
        self.loss_D_GI = self.backward_D_basic(self.netD_G, self.D, fake_IG)

        """Calculate GAN loss for discriminator D_IG (D to B)"""
        fake_GI = self.fake_pools["fake_GI_pool"].query(self.fake_GI)
        self.loss_D_IG = self.backward_D_basic(self.netD_I, self.B, fake_GI)


        #Philips C (H)
        """Calculate GAN loss for discriminator D_HI (C to D)"""
        fake_IH = self.fake_pools["fake_IH_pool"].query(self.fake_IH)
        self.loss_D_HI = self.backward_D_basic(self.netD_H, self.D, fake_IH)

        """Calculate GAN loss for discriminator D_IH (D to C)"""
        fake_HI = self.fake_pools["fake_HI_pool"].query(self.fake_HI)
        self.loss_D_IH = self.backward_D_basic(self.netD_I, self.C, fake_HI)


        self.loss_D = self.loss_D_AG + self.loss_D_GA + self.loss_D_AH + self.loss_D_HA + self.loss_D_AI + self.loss_D_IA + \
                      self.loss_D_BG + self.loss_D_GB + self.loss_D_BH + self.loss_D_HB + self.loss_D_BI + self.loss_D_IB + \
                      self.loss_D_CG + self.loss_D_GC + self.loss_D_CH + self.loss_D_HC + self.loss_D_CI + self.loss_D_IC + \
                      self.loss_D_DG + self.loss_D_GD + self.loss_D_DH + self.loss_D_HD + self.loss_D_DI + self.loss_D_ID + \
                      self.loss_D_EG + self.loss_D_GE + self.loss_D_EH + self.loss_D_HE + self.loss_D_EI + self.loss_D_IE + \
                      self.loss_D_FG + self.loss_D_GF + self.loss_D_FH + self.loss_D_HF + self.loss_D_FI + self.loss_D_IF + \
                      self.loss_D_GH + self.loss_D_HG + self.loss_D_GI + self.loss_D_IG + \
                      self.loss_D_HI + self.loss_D_IH 
             

        return self.loss_D

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
    
 
        # Least squares loss for all the generators
        #B50f
        self.loss_G_AG = self.criterionGAN(self.netD_A(self.fake_GA), True)
        self.loss_G_GA = self.criterionGAN(self.netD_G(self.fake_AG), True)
        self.loss_G_AH = self.criterionGAN(self.netD_A(self.fake_HA), True)
        self.loss_G_HA = self.criterionGAN(self.netD_H(self.fake_AH), True)
        self.loss_G_AI = self.criterionGAN(self.netD_A(self.fake_IA), True)
        self.loss_G_IA = self.criterionGAN(self.netD_I(self.fake_AI), True)

        #B30f
        self.loss_G_BG = self.criterionGAN(self.netD_B(self.fake_GB), True)
        self.loss_G_GB = self.criterionGAN(self.netD_G(self.fake_BG), True)
        self.loss_G_BH = self.criterionGAN(self.netD_B(self.fake_HB), True)
        self.loss_G_HB = self.criterionGAN(self.netD_H(self.fake_BH), True)
        self.loss_G_BI = self.criterionGAN(self.netD_B(self.fake_IB), True)
        self.loss_G_IB = self.criterionGAN(self.netD_I(self.fake_BI), True)

        #GE BONE
        self.loss_G_CG = self.criterionGAN(self.netD_C(self.fake_GC), True)
        self.loss_G_GC = self.criterionGAN(self.netD_G(self.fake_CG), True)
        self.loss_G_CH = self.criterionGAN(self.netD_C(self.fake_HC), True)
        self.loss_G_HC = self.criterionGAN(self.netD_H(self.fake_CH), True)
        self.loss_G_CI = self.criterionGAN(self.netD_C(self.fake_IC), True)
        self.loss_G_IC = self.criterionGAN(self.netD_I(self.fake_CI), True)

        #GE STD
        self.loss_G_DG = self.criterionGAN(self.netD_D(self.fake_GD), True)
        self.loss_G_GD = self.criterionGAN(self.netD_G(self.fake_DG), True)
        self.loss_G_DH = self.criterionGAN(self.netD_D(self.fake_HD), True)
        self.loss_G_HD = self.criterionGAN(self.netD_H(self.fake_DH), True)
        self.loss_G_DI = self.criterionGAN(self.netD_D(self.fake_ID), True)
        self.loss_G_ID = self.criterionGAN(self.netD_I(self.fake_DI), True)

        #GE LUNG
        self.loss_G_EG = self.criterionGAN(self.netD_E(self.fake_GE), True)
        self.loss_G_GE = self.criterionGAN(self.netD_G(self.fake_EG), True) 
        self.loss_G_EH = self.criterionGAN(self.netD_E(self.fake_HE), True)
        self.loss_G_HE = self.criterionGAN(self.netD_H(self.fake_EH), True)
        self.loss_G_EI = self.criterionGAN(self.netD_E(self.fake_IE), True)
        self.loss_G_IE = self.criterionGAN(self.netD_I(self.fake_EI), True)

        #B80f 
        self.loss_G_FG = self.criterionGAN(self.netD_F(self.fake_GF), True)
        self.loss_G_GF = self.criterionGAN(self.netD_G(self.fake_FG), True)
        self.loss_G_FH = self.criterionGAN(self.netD_F(self.fake_HF), True)
        self.loss_G_HF = self.criterionGAN(self.netD_H(self.fake_FH), True)
        self.loss_G_FI = self.criterionGAN(self.netD_F(self.fake_IF), True)
        self.loss_G_IF = self.criterionGAN(self.netD_I(self.fake_FI), True)

        #Philips B 
        self.loss_G_GH = self.criterionGAN(self.netD_G(self.fake_HG), True)
        self.loss_G_HG = self.criterionGAN(self.netD_H(self.fake_GH), True)
        self.loss_G_GI = self.criterionGAN(self.netD_G(self.fake_IG), True)
        self.loss_G_IG = self.criterionGAN(self.netD_I(self.fake_GI), True)

        #Philips C
        self.loss_G_HI = self.criterionGAN(self.netD_H(self.fake_IH), True)
        self.loss_G_IH = self.criterionGAN(self.netD_I(self.fake_HI), True)

        #Multipath cycleGAN: Cycle consistency losses
        #B50f
        self.loss_cycle_AG = self.criterionCycle(self.rec_AG, self.B50f) * lambda_A
        self.loss_cycle_GA = self.criterionCycle(self.rec_GA, self.B) * lambda_B
        self.loss_cycle_AH = self.criterionCycle(self.rec_AH, self.B50f) * lambda_A
        self.loss_cycle_HA = self.criterionCycle(self.rec_HA, self.C) * lambda_B
        self.loss_cycle_AI = self.criterionCycle(self.rec_AI, self.B50f) * lambda_A
        self.loss_cycle_IA = self.criterionCycle(self.rec_IA, self.D) * lambda_B

        #B30f
        self.loss_cycle_BG = self.criterionCycle(self.rec_BG, self.B30f) * lambda_A
        self.loss_cycle_GB = self.criterionCycle(self.rec_GB, self.B) * lambda_B
        self.loss_cycle_BH = self.criterionCycle(self.rec_BH, self.B30f) * lambda_A
        self.loss_cycle_HB = self.criterionCycle(self.rec_HB, self.C) * lambda_B
        self.loss_cycle_BI = self.criterionCycle(self.rec_BI, self.B30f) * lambda_A
        self.loss_cycle_IB = self.criterionCycle(self.rec_IB, self.D) * lambda_B

        #GE BONE
        self.loss_cycle_CG = self.criterionCycle(self.rec_CG, self.BONE) * lambda_A
        self.loss_cycle_GC = self.criterionCycle(self.rec_GC, self.B) * lambda_B
        self.loss_cycle_CH = self.criterionCycle(self.rec_CH, self.BONE) * lambda_A
        self.loss_cycle_HC = self.criterionCycle(self.rec_HC, self.C) * lambda_B
        self.loss_cycle_CI = self.criterionCycle(self.rec_CI, self.BONE) * lambda_A
        self.loss_cycle_IC = self.criterionCycle(self.rec_IC, self.D) * lambda_B

        #STD
        self.loss_cycle_DG = self.criterionCycle(self.rec_DG, self.STD) * lambda_A
        self.loss_cycle_GD = self.criterionCycle(self.rec_GD, self.B) * lambda_B
        self.loss_cycle_DH = self.criterionCycle(self.rec_DH, self.STD) * lambda_A
        self.loss_cycle_HD = self.criterionCycle(self.rec_HD, self.C) * lambda_B
        self.loss_cycle_DI = self.criterionCycle(self.rec_DI, self.STD) * lambda_A
        self.loss_cycle_ID = self.criterionCycle(self.rec_ID, self.D) * lambda_B

        #LUNG
        self.loss_cycle_EG = self.criterionCycle(self.rec_EG, self.LUNG) * lambda_A
        self.loss_cycle_GE = self.criterionCycle(self.rec_GE, self.B) * lambda_B
        self.loss_cycle_EH = self.criterionCycle(self.rec_EH, self.LUNG) * lambda_A
        self.loss_cycle_HE = self.criterionCycle(self.rec_HE, self.C) * lambda_B
        self.loss_cycle_EI = self.criterionCycle(self.rec_EI, self.LUNG) * lambda_A
        self.loss_cycle_IE = self.criterionCycle(self.rec_IE, self.D) * lambda_B

        #B80f 
        self.loss_cycle_FG = self.criterionCycle(self.rec_FG, self.B80f) * lambda_A
        self.loss_cycle_GF = self.criterionCycle(self.rec_GF, self.B) * lambda_B
        self.loss_cycle_FH = self.criterionCycle(self.rec_FH, self.B80f) * lambda_A
        self.loss_cycle_HF = self.criterionCycle(self.rec_HF, self.C) * lambda_B
        self.loss_cycle_FI = self.criterionCycle(self.rec_FI, self.B80f) * lambda_A
        self.loss_cycle_IF = self.criterionCycle(self.rec_IF, self.D) * lambda_B

        #Philips B 
        self.loss_cycle_GH = self.criterionCycle(self.rec_GH, self.B) * lambda_A
        self.loss_cycle_HG = self.criterionCycle(self.rec_HG, self.C) * lambda_B
        self.loss_cycle_GI = self.criterionCycle(self.rec_GI, self.B) * lambda_A
        self.loss_cycle_IG = self.criterionCycle(self.rec_IG, self.D) * lambda_B

        #Philips C
        self.loss_cycle_HI = self.criterionCycle(self.rec_HI, self.C) * lambda_A
        self.loss_cycle_IH = self.criterionCycle(self.rec_IH, self.D) * lambda_B
       
        
        #Tissue statistic loss
        #B50f
        self.loss_segB50fPhilB, self.loss_segPhilBB50f = self.tissue_statistic_loss(self.B50f, self.fake_AG, self.B, self.fake_GA, self.B50f_mask, self.B_mask)
        self.loss_segB50fPhilC, self.loss_segPhilCB50f = self.tissue_statistic_loss(self.B50f, self.fake_AH, self.C, self.fake_HA, self.B50f_mask, self.C_mask)
        self.loss_segB50fPhilD, self.loss_segPhilDB50f = self.tissue_statistic_loss(self.B50f, self.fake_AI, self.D, self.fake_IA, self.B50f_mask, self.D_mask)

        #B30f
        self.loss_segB30fPhilB, self.loss_segPhilBB30f = self.tissue_statistic_loss(self.B30f, self.fake_BG, self.B, self.fake_GB, self.B30f_mask, self.B_mask)
        self.loss_segB30fPhilC, self.loss_segPhilCB30f = self.tissue_statistic_loss(self.B30f, self.fake_BH, self.C, self.fake_HB, self.B30f_mask, self.C_mask)
        self.loss_segB30fPhilD, self.loss_segPhilDB30f = self.tissue_statistic_loss(self.B30f, self.fake_BI, self.D, self.fake_IB, self.B30f_mask, self.D_mask)

        #BONE
        self.loss_segBONEPhilB, self.loss_segPhilBBONE = self.tissue_statistic_loss(self.BONE, self.fake_CG, self.B, self.fake_GC, self.BONE_mask, self.B_mask)
        self.loss_segBONEPhilC, self.loss_segPhilCBONE = self.tissue_statistic_loss(self.BONE, self.fake_CH, self.C, self.fake_HC, self.BONE_mask, self.C_mask)
        self.loss_segBONEPhilD, self.loss_segPhilDBONE = self.tissue_statistic_loss(self.BONE, self.fake_CI, self.D, self.fake_IC, self.BONE_mask, self.D_mask)

        #STD
        self.loss_segSTDPhilB, self.loss_segPhilBSTD = self.tissue_statistic_loss(self.STD, self.fake_DG, self.B, self.fake_GD, self.STD_mask, self.B_mask)
        self.loss_segSTDPhilC, self.loss_segPhilCSTD = self.tissue_statistic_loss(self.STD, self.fake_DH, self.C, self.fake_HD, self.STD_mask, self.C_mask)
        self.loss_segSTDPhilD, self.loss_segPhilDSTD = self.tissue_statistic_loss(self.STD, self.fake_DI, self.D, self.fake_ID, self.STD_mask, self.D_mask)

        #LUNG 
        self.loss_segLUNGPhilB, self.loss_segPhilBLUNG = self.tissue_statistic_loss(self.LUNG, self.fake_EG, self.B, self.fake_GE, self.LUNG_mask, self.B_mask)
        self.loss_segLUNGPhilC, self.loss_segPhilCLUNG = self.tissue_statistic_loss(self.LUNG, self.fake_EH, self.C, self.fake_HE, self.LUNG_mask, self.C_mask)
        self.loss_segLUNGPhilD, self.loss_segPhilDLUNG = self.tissue_statistic_loss(self.LUNG, self.fake_EI, self.D, self.fake_IE, self.LUNG_mask, self.D_mask)

        #B80f
        self.loss_segB80fPhilB, self.loss_segPhilBB80f = self.tissue_statistic_loss(self.B80f, self.fake_FG, self.B, self.fake_GF, self.B80f_mask, self.B_mask)
        self.loss_segB80fPhilC, self.loss_segPhilCB80f = self.tissue_statistic_loss(self.B80f, self.fake_FH, self.C, self.fake_HF, self.B80f_mask, self.C_mask)
        self.loss_segB80fPhilD, self.loss_segPhilDB80f = self.tissue_statistic_loss(self.B80f, self.fake_FI, self.D, self.fake_IF, self.B80f_mask, self.D_mask)

        #Philips B 
        self.loss_segPhilBPhilC, self.loss_segPhilCPhilB = self.tissue_statistic_loss(self.B, self.fake_GH, self.C, self.fake_HG, self.B_mask, self.C_mask)
        self.loss_segPhilBPhilD, self.loss_segPhilDPhilB = self.tissue_statistic_loss(self.B, self.fake_GI, self.D, self.fake_IG, self.B_mask, self.D_mask)
       
       #Philips C 
        self.loss_segPhilCPhilD, self.loss_segPhilDPhilC = self.tissue_statistic_loss(self.C, self.fake_HI, self.D, self.fake_IH, self.C_mask, self.D_mask)

        #this is loss function for multipath cycleGAN: Adversarial losses + desicriminator losses + Seg losses
        self.loss_G =  self.loss_G_AG + self.loss_G_GA + self.loss_G_AH + self.loss_G_HA + self.loss_G_AI + self.loss_G_IA + \
                       self.loss_G_BG + self.loss_G_GB + self.loss_G_BH + self.loss_G_HB + self.loss_G_BI + self.loss_G_IB + \
                       self.loss_G_CG + self.loss_G_GC + self.loss_G_CH + self.loss_G_HC + self.loss_G_CI + self.loss_G_IC + \
                       self.loss_G_DG + self.loss_G_GD + self.loss_G_DH + self.loss_G_HD + self.loss_G_DI + self.loss_G_ID + \
                       self.loss_G_EG + self.loss_G_GE + self.loss_G_EH + self.loss_G_HE + self.loss_G_EI + self.loss_G_IE + \
                       self.loss_G_FG + self.loss_G_GF + self.loss_G_FH + self.loss_G_HF + self.loss_G_FI + self.loss_G_IF + \
                       self.loss_G_GH + self.loss_G_HG + self.loss_G_GI + self.loss_G_IG + self.loss_G_HI + self.loss_G_IH + \
                       self.loss_cycle_AG + self.loss_cycle_GA + self.loss_cycle_AH + self.loss_cycle_HA + self.loss_cycle_AI + self.loss_cycle_IA + \
                       self.loss_cycle_BG + self.loss_cycle_GB + self.loss_cycle_BH + self.loss_cycle_HB + self.loss_cycle_BI + self.loss_cycle_IB + \
                       self.loss_cycle_CG + self.loss_cycle_GC + self.loss_cycle_CH + self.loss_cycle_HC + self.loss_cycle_CI + self.loss_cycle_IC + \
                       self.loss_cycle_DG + self.loss_cycle_GD + self.loss_cycle_DH + self.loss_cycle_HD + self.loss_cycle_DI + self.loss_cycle_ID + \
                       self.loss_cycle_EG + self.loss_cycle_GE + self.loss_cycle_EH + self.loss_cycle_HE + self.loss_cycle_EI + self.loss_cycle_IE + \
                       self.loss_cycle_FG + self.loss_cycle_GF + self.loss_cycle_FH + self.loss_cycle_HF + self.loss_cycle_FI + self.loss_cycle_IF + \
                       self.loss_cycle_GH + self.loss_cycle_HG + self.loss_cycle_GI + self.loss_cycle_IG + self.loss_cycle_HI + self.loss_cycle_IH + \
                       self.loss_segB50fPhilB + self.loss_segPhilBB50f + self.loss_segB50fPhilC + self.loss_segPhilCB50f + self.loss_segB50fPhilD + self.loss_segPhilDB50f + \
                       self.loss_segB30fPhilB + self.loss_segPhilBB30f + self.loss_segB30fPhilC + self.loss_segPhilCB30f + self.loss_segB30fPhilD + self.loss_segPhilDB30f + \
                       self.loss_segBONEPhilB + self.loss_segPhilBBONE + self.loss_segBONEPhilC + self.loss_segPhilCBONE + self.loss_segBONEPhilD + self.loss_segPhilDBONE + \
                       self.loss_segSTDPhilB + self.loss_segPhilBSTD + self.loss_segSTDPhilC + self.loss_segPhilCSTD + self.loss_segSTDPhilD + self.loss_segPhilDSTD + \
                       self.loss_segLUNGPhilB + self.loss_segPhilBLUNG + self.loss_segLUNGPhilC + self.loss_segPhilCLUNG + self.loss_segLUNGPhilD + self.loss_segPhilDLUNG + \
                       self.loss_segB80fPhilB + self.loss_segPhilBB80f + self.loss_segB80fPhilC + self.loss_segPhilCB80f + self.loss_segB80fPhilD + self.loss_segPhilDB80f + \
                       self.loss_segPhilBPhilC + self.loss_segPhilCPhilB + self.loss_segPhilBPhilD + self.loss_segPhilDPhilB + \
                       self.loss_segPhilCPhilD + self.loss_segPhilDPhilC
                            

        return self.loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F, self.netD_G, self.netD_H, self.netD_I], False) #Multipath
        self.optimizer_G.zero_grad()
        with autocast():
            loss_G = self.backward_G()          # calculate gradients for all G's
        self.scalar.scale(loss_G).backward() 
        self.scalar.step(self.optimizer_G)
        self.scalar.update()

        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C, self.netD_D, self.netD_E, self.netD_F, self.netD_G, self.netD_H, self.netD_I], True) #Multipath
        self.optimizer_D.zero_grad()
        with autocast():
            loss_D = self.backward_D()      # calculate gradients for all D's
        self.scalar.scale(loss_D).backward()
        self.scalar.step(self.optimizer_D)
        self.scalar.update()