import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch import nn
import torch 
import os 
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import wandb


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_everything(123)
    opt = TrainOptions().parse()   
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)    #5000 images per epoch
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      
    model.setup(opt)       
    visualizer = Visualizer(opt)   
    total_iters = 0                # the total number of training iterations

    #Track losses with wandb
    wandb.init(
        project="Stage1_NLST_MultipathCycleGAN_with_Anatomy_guidance_stage2",
        config= {
            "batch_size": opt.batch_size,
            "lambda_seg": opt.lambda_seg,
            "lr": opt.lr,
            "n_epochs": opt.n_epochs,
            "n_epochs_decay": opt.n_epochs_decay,
            "dataset_size": dataset_size
        }
    )

    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
       
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0                  
        visualizer.reset()              
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(tqdm(dataset)):

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                wandb.log({k: v for k, v in losses.items()}) #Log losses onto weights and biases
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))