import math
import os.path
import argparse

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from datetime import datetime
from tqdm import *

from steve import STEVE
from data import GlobVideoDataset, SAYCAMDataset
from utils import cosine_anneal, linear_warmup

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print("CUDA available:", torch.cuda.is_available())
# print("Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=8)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', default='data/*')
parser.add_argument('--log_path', default='/scratch/wz3008/cvcl-related/steve_logs')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--steps', type=int, default=200000)

parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=15)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_predictor_blocks', type=int, default=1)
parser.add_argument('--num_predictor_heads', type=int, default=4)
parser.add_argument('--predictor_dropout', type=int, default=0.0)

parser.add_argument('--vocab_size', type=int, default=2048)
parser.add_argument('--num_decoder_blocks', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')
parser.add_argument('--use_dp', default=False, action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

# train_dataset = GlobVideoDataset(root=args.data_path, phase='train', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
# val_dataset = GlobVideoDataset(root=args.data_path, phase='val', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
train_dataset = SAYCAMDataset("saycam_transcript_5fps","/home/wz3008/steve/", "train",128)
val_dataset = SAYCAMDataset("saycam_transcript_5fps","/home/wz3008/steve/", "val",128)

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

model = STEVE(args)

if os.path.isfile(args.checkpoint_path):
    print(f"Using {args.checkpoint_path}:")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

for param in model.backbone.parameters():
    param.requires_grad = False

if args.use_dp:
    model = DP(model)
    optimizer = Adam([
        {'params': [p for n,p in model.module.named_parameters() if 'dvae' in n or 'deconvDVAE' in n], 'lr': args.lr_dvae},
        {'params': [p for n,p in model.module.named_parameters() if 'steve_encoder' in n or 'deconvCNN' in n], 'lr': args.lr_enc},
        {'params': [p for n,p in model.module.named_parameters() if 'steve_decoder' in n], 'lr': args.lr_dec},
    ])
else:
    optimizer = Adam([
        {'params': [p for n,p in model.named_parameters() if 'dvae' in n or 'deconvDVAE' in n], 'lr': args.lr_dvae},
        {'params': [p for n,p in model.named_parameters() if 'steve_encoder' in n or 'deconvCNN' in n], 'lr': args.lr_enc},
        {'params': [p for n,p in model.named_parameters() if 'steve_decoder' in n], 'lr': args.lr_dec},
    ])
model.to(device)

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

def visualize(video, recon_dvae, recon_tf, attns, N=8):
    B, T, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        # recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        # recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :]

        # tile
        # tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(end_dim=1)
        tiles = torch.cat((video_t, attns_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(args.num_slots + 1), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames

out_dir = "/home/wz3008/steve/visualize_frames"
os.makedirs(out_dir, exist_ok=True)
print("<=== Training Start ===>")
print(f"start_epoch: {start_epoch}")
for epoch in tqdm(range(start_epoch, args.epochs)):
    model.train()
    
    for batch, video in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch

        tau = cosine_anneal(global_step, args.tau_start, args.tau_final, 0, args.tau_steps)

        lr_warmup_factor_enc = linear_warmup(global_step, 0., 1.0, 0., args.lr_warmup_steps)

        lr_warmup_factor_dec = linear_warmup(global_step, 0., 1.0, 0, args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

        video = video.to(device)

        optimizer.zero_grad()
        
        (_, cross_entropy, _, attns) = model(video, tau, args.hard)

        if args.use_dp:
            # mse = mse.mean()
            cross_entropy = cross_entropy.mean()

        loss = cross_entropy
        
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip, 'inf')
        optimizer.step()
        
        with torch.no_grad():
            if batch % log_interval == 0:
                # print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                #       epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                # writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                # writer.add_scalar('TRAIN/mse', mse.item(), global_step)

                # writer.add_scalar('TRAIN/tau', tau, global_step)
                # writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                # writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                # writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

    
    # with torch.no_grad():
    #     frames = visualize(video, None, None, attns, N=8)
    #     # writer.add_video('TRAIN_recons/epoch={:03}'.format(epoch+1), frames)
    #     for i, img in enumerate(frames):
    #         save_image(img, os.path.join(out_dir, f"epoch={(epoch+1):03}_frame_{i:02d}.png"))
    #         print(f"sve epoch={(epoch+1):03}_frame_{i:02d}.png to "+out_dir)
    
    with torch.no_grad():
        model.eval()

        val_cross_entropy = 0.
        # val_mse = 0.

        for batch, video in enumerate(val_loader):
            video = video.to(device)

            (_, cross_entropy, _, attns) = model(video, tau, args.hard)

            if args.use_dp:
                # mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            val_cross_entropy += cross_entropy.item()
            # val_mse += mse.item()

        val_cross_entropy /= (val_epoch_size)
        # val_mse /= (val_epoch_size)

        val_loss = val_cross_entropy

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        # writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
        # writer.add_scalar('VAL/mse', val_mse, epoch+1)

        print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch+1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

            # if global_step < args.steps:
            #     torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), os.path.join(log_dir, f'best_model_until_{args.steps}_steps_at_{epoch}.pt'))

            if 50 <= epoch:
                # gen_video = (model.module if args.use_dp else model).reconstruct_autoregressive(video[:8])
                # frames = visualize(video, recon, gen_video, attns, N=8)
                frames = visualize(video, None, None, attns, N=8)
                # writer.add_video('VAL_recons/epoch={:03}'.format(epoch + 1), frames)
                for i, img in enumerate(frames):
                    save_image(img, os.path.join(out_dir, f"epoch={(epoch+1):03}_frame_{i:02d}.png"))
                    # print(f"sve epoch={(epoch+1):03}_frame_{i:02d}.png to "+out_dir)
                
        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict() if args.use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
