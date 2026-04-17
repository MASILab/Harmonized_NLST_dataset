import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import itertools


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        print(self.device)
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.newstage_train = opt.stage2_checkpoints #This will point to the .pth file for the best epoch from stage 1 training

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:                                                                                                                                                                                                                                                        
             self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]                                                                                                                                                                         
        if not self.isTrain or opt.continue_train: #Load model weights to continue training                                                                                                                                                                                     
             load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch                                                                                                                                                                                         
             self.load_networks(load_suffix)                                                                                                                                                                                                                                     
        elif self.isTrain and opt.stage == 2: #Load weights of older domains to train stage 2                                                                                                                                                                                   
             load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch                                                                                                                                                                                         
             self.load_partial_networks(load_suffix)                                                                                                                                                                                                                             
        elif self.isTrain and opt.continue_train_stage2 and opt.stage == 2:                                                                                                                                                                                                     
             #Load the weights of all the models from the epoch                                                                                                                                                                                                                  
             load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch                                                                                                                                                                                         
             self.load_stage2_networks(load_suffix)                                                                                                                                                                                                                              
        self.print_networks(opt.verbose)  

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        model_state_dicts = {}
        for name in self.model_names:
            if isinstance(name, str):
                # save_filename = '%s_net_%s.pth' % (epoch, name)
                # save_path = os.path.join(self.save_dir, save_filename)
                # save_opt_path = os.path.join(self.save_dir, save_optimizer)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    model_state_dicts[name] = net.module.cpu().state_dict()
                    net.cuda(self.gpu_ids[0])
                else:
                    # torch.save(net.cpu().state_dict(), save_path)
                    model_state_dicts[name] = net.cpu().state_dict()
        
        save_filename = '%s_net_gendisc_weights.pth' % (epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(model_state_dicts, save_path)

        #save the D and G optimizer states 
        optimizer = '%s_optimizer.pth' % (epoch)
        save_opt_path = os.path.join(self.save_dir, optimizer)
        torch.save({'epoch': epoch, 'G_optimizer': self.optimizer_G.state_dict(), 'D_optimizer': self.optimizer_D.state_dict()}, save_opt_path)
        


    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
                
        
    def load_networks(self, epoch):
        """Load all the networks from the disk. Use this function when all models are pretrained.

        Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        load_filename = '%s_net_gendisc_weights.pth' % (epoch)
        load_path = os.path.join(self.save_dir, load_filename)
        print('loading the model dictionary from %s' % load_path)
        state_dicts = torch.load(load_path, map_location=str(self.device))

        for name in self.model_names:
            if isinstance(name, str): 
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model %s from dictionary' % name)
                state_dict = state_dicts[name]
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata


                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
        

        opt_filename = '%s_optimizer_%s.pth' % (epoch)
        load_opt_path = os.path.join(self.save_dir, opt_filename)
        print('Loading the optimizer dictionary from %s' % load_opt_path)
        optimizer_state_dicts = torch.load(load_opt_path, map_location=str(self.device))

        self.optimizer_G.load_state_dict(optimizer_state_dicts['G_optimizer'])
        self.optimizer_D.load_state_dict(optimizer_state_dicts['D_optimizer'])
    
    
    def load_stage2_networks(self, epoch): #Loading for continuing stage 2 training
        load_filename = '%s_net_gendisc_weights.pth' % (epoch)
        load_path = os.path.join(self.newstage_train, load_filename)
        print('loading the model dictionary from %s' % load_path)
        state_dicts = torch.load(load_path, map_location=str(self.device))

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model %s from dictionary' % name)
                state_dict = state_dicts[name]
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

            # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
        
        opt_filename = '%s_optimizer_dict.pth' % epoch
        opt_path = os.path.join(self.save_dir, opt_filename)
        print('loading the optimizer dictionary from %s' % opt_path)
        optimizer_state_dicts = torch.load(opt_path, map_location=str(self.device))
    
        self.optimizer_G.load_state_dict(optimizer_state_dicts['G_optimizer'])
        self.optimizer_D.load_state_dict(optimizer_state_dicts['D_optimizer'])
    

    def load_model_weights(self, net_name, state_dicts): #Helper function
        if isinstance(net_name, str):
            net = getattr(self, 'net' + net_name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            state_dict = state_dicts[net_name]
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            for key in list(state_dict.keys()):
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)



    def load_partial_networks(self, epoch): 
        """
        Loading pretrained networks from stage 1 to train new paths in stage 2
        """
        pretrained_nets = ['G_B50f_encoder', 'G_B50f_decoder', 'G_B30f_encoder', 'G_B30f_decoder',
                           'G_BONE_encoder', 'G_BONE_decoder', 'G_STD_encoder', 'G_STD_decoder',
                           'G_LUNG_encoder', 'G_LUNG_decoder', 'G_B80f_encoder', 'G_B80f_decoder',
                           'D_A', 'D_B', 'D_C', 'D_D', 'D_E', 'D_F']
        
        load_filename = '%s_net_gendisc_weights.pth' % (epoch)
        load_path = os.path.join(self.newstage_train, load_filename)
        print('loading the model dictionary for pretrained nets from %s' % load_path)
        state_dicts = torch.load(load_path, map_location=str(self.device))
        
        for name in self.model_names:
            if name in pretrained_nets and name in state_dicts:
                print(f"Loading weights of {name} models from the previous training!")
                self.load_model_weights(name, state_dicts)
            else: 
                if name.endswith("encoder") or name.endswith("decoder"):
                    print(f"Loading average weights of pretrained models for {name}")
                    model_name = name.split("_")[2]
                    statedict = self.average_pretrained_weights(model_name) #State dict of average weights
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    if hasattr(statedict, '_metadata'):
                        del statedict._metadata
                    for key in list(statedict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(statedict, net, key.split('.'))
                    net.load_state_dict(statedict)
                elif name.startswith("D"):
                    model_name = "discriminator"
                    statedict = self.average_pretrained_weights(model_name)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    if hasattr(statedict, '_metadata'):
                        del statedict._metadata
                    for key in list(statedict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(statedict, net, key.split('.'))
                    net.load_state_dict(statedict)
                    print(f"{name} is a new discriminator. Initialize with average weights of pretrained discriminators")

               

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    

    def average_pretrained_weights(self, model_type):
        """
        Initialize the weights of a new encoder/decoder using the average of the weights of the pretrained encoders/decoders. Epoch 100 was the best for stage 1.
        """

        checkpoint = os.path.join(self.newstage_train, "152_net_gendisc_weights.pth")

        if model_type == "encoder":
            m1 = torch.load([checkpoint]['G_B50f_encoder'])
            m2 = torch.load([checkpoint]['G_B30f_encoder'])
            m3 = torch.load([checkpoint]['G_BONE_encoder'])
            m4 = torch.load([checkpoint]['G_STD_encoder'])
            m5 = torch.load([checkpoint]['G_LUNG_encoder'])
            m6 = torch.load([checkpoint]['G_B80f_encoder'])
        elif model_type == "decoder":
            m1 = torch.load([checkpoint]['G_B50f_decoder'])
            m2 = torch.load([checkpoint]['G_B30f_decoder'])
            m3 = torch.load([checkpoint]['G_BONE_decoder'])
            m4 = torch.load([checkpoint]['G_STD_decoder'])
            m5 = torch.load([checkpoint]['G_LUNG_decoder'])
            m6 = torch.load([checkpoint]['G_B80f_decoder'])
        elif model_type == 'discriminator':
            m1 = torch.load([checkpoint]['D_A'])
            m2 = torch.load([checkpoint]['D_B'])
            m3 = torch.load([checkpoint]['D_C'])
            m4 = torch.load([checkpoint]['D_D'])
            m5 = torch.load([checkpoint]['D_E'])
            m6 = torch.load([checkpoint]['D_F'])

        M_avg = OrderedDict()
        for k in m1.keys():
            M_avg[k] = (m1[k] + m2[k] + m3[k] + m4[k] + m5[k] + m6[k]) / 6.0
        return M_avg