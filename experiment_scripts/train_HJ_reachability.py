'''Reproduces Paper Sec. 4.3 and Supplement Sec. 5'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse
import seaborn as sb
import matplotlib.pyplot as plt
import torch
import numpy as np

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--velocity', type=str, default='uniform', required=False, choices=['uniform', 'square', 'circle'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet and neumann conditions')
p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--load_model', type=str, default=None, required=False,
               help='Load pretrained model from checkpoint.')
p.add_argument('--play', action='store_true', default=False, required=False, help='plot the predicted results?')

opt = p.parse_args()

print('opt:', opt)

# if we have a velocity perturbation, offset the source
source_coords = [0., 0., 0.]

#dataset = dataio.HJReachability(sidelength=48, pretrain=opt.pretrain)
dataset = dataio.HJReachability(sidelength=12, pretrain=opt.pretrain) # GTX1050

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=4, out_features=1, type=opt.model, mode=opt.mode,
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=1)

if opt.load_model is not None:
    model.load_state_dict(torch.load(opt.load_model))

model.cuda()

print('opt.play:', opt.play)

if not opt.play:
    # Define the loss
    loss_fn = loss_functions.wave_HJ_reachability
    summary_fn = utils.write_HJ_reachability_summary # CHECK!!!!!!!!!!

    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=opt.clip_grad,
                use_lbfgs=opt.use_lbfgs)

device = torch.device("cuda:0")
dtype = torch.float
with torch.no_grad():
    T = 1 # plot timestamp, range [0,1]
    step_eval = 101
    x1_eval = torch.linspace(-1, 1, steps = step_eval, device=device, dtype=dtype).view(-1,1)
    x1_eval = x1_eval.repeat(step_eval,1)
    x2_eval = torch.linspace(-1, 1, steps = step_eval, device=device, dtype=dtype).view(-1,1)
    x2_eval = x2_eval.repeat_interleave(step_eval, dim = 0)
    x3_eval = np.pi/2 * torch.ones((step_eval * step_eval, 1), device=device, dtype=dtype) # fixed x3 = pi/2
    x3_eval = x3_eval / np.pi # normalization
    t_eval = T * torch.ones((step_eval * step_eval, 1), device=device, dtype=dtype) # fixed t = T
    xt_eval = torch.cat([t_eval, x1_eval, x2_eval, x3_eval], dim = 1)
    #print('xt_eval:', xt_eval)
    V_eval = model({'coords':xt_eval})['model_out'].view(step_eval, step_eval)
    V0 = model({'coords':torch.tensor([[T, 0, 0, 0.5]], device=device, dtype=dtype)})['model_out']
    print('V(T, 0,0,pi/2)=', V0.item())
    V1 = model({'coords':torch.tensor([[T, 1, 1, 0.5]], device=device, dtype=dtype)})['model_out']
    print('V(T, 1,1,pi/2)=', V1.item())
    V2 = model({'coords':torch.tensor([[T, 1, -1, 0.5]], device=device, dtype=dtype)})['model_out']
    print('V(T, 1,-1,pi/2)=', V2.item())    
    V3 = model({'coords':torch.tensor([[T, -1, 1, 0.5]], device=device, dtype=dtype)})['model_out']
    print('V(T, -1,1,pi/2)=', V3.item()) 
    V4 = model({'coords':torch.tensor([[T, -1, -1, 0.5]], device=device, dtype=dtype)})['model_out']
    print('V(T, -1,-1,pi/2)=', V4.item()) 

    heat_map = sb.heatmap(V_eval.cpu().numpy())
    heat_map.invert_yaxis()
    # heat_map.set_yticks(np.linspace(-1, 1, 11) * heat_map.get_ylim()[1])
    plt.show()
