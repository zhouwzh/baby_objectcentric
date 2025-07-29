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

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')  # Force to use CPU for debugging

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
parser.add_argument('--local', default=False, action='store_true')
parser.add_argument('--dev', default=False, action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

if not args.local:
    train_dataset = SAYCAMDataset("saycam_transcript_5fps","/home/wz3008/steve/", "train",128)
    val_dataset = SAYCAMDataset("saycam_transcript_5fps","/home/wz3008/steve/", "val",128)
else:
    DATA_DIR = "/mnt/wwn-0x5000c500e421004a/yy2694/datasets/train_5fps"
    train_dataset = SAYCAMDataset(img_dir=DATA_DIR,json_path="/home/wz3008/steve/", phase="train",img_size=128)
    val_dataset = SAYCAMDataset(img_dir=DATA_DIR,json_path="/home/wz3008/steve/", phase="val",img_size=128)

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


model.to(device)
model.eval()

out_dir = "/home/wz3008/baby_objectcentric/visualize_frames"
os.makedirs(out_dir, exist_ok=True)
print("<=== Training Start ===>")
print(f"start_epoch: {start_epoch}")
for epoch in tqdm(range(start_epoch, args.epochs)):
    for batch, video in enumerate(train_loader):
        model.visual_cvcl(video)
        break  # For debugging, only process one batch per epoch 
    break  # For debugging, only run one epoch  