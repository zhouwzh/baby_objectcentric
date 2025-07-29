from utils import *

from dvae import dVAE
from transformer import TransformerEncoder, TransformerDecoder
from multimodal.multimodal_lit import MultiModalLitModel


class SlotAttentionVideo(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size,
                 num_predictor_blocks=1,
                 num_predictor_heads=4,
                 dropout=0.1,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))
        self.predictor = TransformerEncoder(num_predictor_blocks, slot_size, num_predictor_heads, dropout)

    def forward(self, inputs):
        B, T, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn_vis = F.softmax(attn_logits, dim=-1)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.view(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)

                # use MLP only when more than one iterations
                if i < self.num_iterations - 1:
                    slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

            # predictor
            slots = self.predictor(slots)

        attns_collect = torch.stack(attns_collect, dim=1)   # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)   # B, T, num_slots, slot_size

        return slots_collect, attns_collect


class LearnedPositionalEmbedding1D(nn.Module):

    def __init__(self, num_inputs, input_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, num_inputs, input_size), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, offset=0):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:, offset:offset + T])


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class STEVEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.cnn1 = nn.Sequential(
            Conv2dBlock(args.img_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        self.cnn = nn.Sequential(
            Conv2dBlock(args.vocab_size, args.cnn_hidden_size, 5, 1, 2),    #nn.Conv2d + relu()
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(args.d_model, args.image_size if args.image_size == 64 else args.image_size // 2)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_model, args.d_model))

        self.savi = SlotAttentionVideo(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size,
            args.num_predictor_blocks, args.num_predictor_heads, args.predictor_dropout)

        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)


class STEVEDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)

        self.pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)

        self.tf = TransformerDecoder(
            args.num_decoder_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_decoder_heads, args.dropout)

        self.head = linear(args.d_model, args.vocab_size, bias=False)

class CVCL_VISION_ENCODER(nn.Module):
    def __init__(self):
        super().__init__()
        cvcl, preprocess = MultiModalLitModel.load_model(model_name="cvcl")

        cvcl.vision_encoder.model.maxpool = nn.Identity()  # remove maxpool layer
        blk3 = cvcl.vision_encoder.model.layer3[0]
        blk3.conv2.stride = (1, 1)  # change stride to 1 for layer3
        blk3.downsample[0].stride = (1, 1)  # change stride to 1 for downsample in layer3
        blk4 = cvcl.vision_encoder.model.layer4[0]
        blk4.conv2.stride = (1, 1)  # change stride to 1 for layer4
        blk4.downsample[0].stride = (1, 1)  # change stride to 1 for downsample in layer4

        self.cvcl = cvcl
        self.preprocess = preprocess
    def forward(self, x):
        features = {}
        def hook_fn(module, input, output):
            features['layer4'] = output
        hook = self.cvcl.vision_encoder.model.layer4.register_forward_hook(hook_fn)
        image_features = self.cvcl.encode_image(x)
        hook.remove()

        return features['layer4']

class TransitionModule(nn.Module): #to B*T, d_model, H/2, W/2
    def __init__(self, args):
        super().__init__()
        self.channel_adjust = nn.Conv2d(2048, args.d_model, kernel_size=1)
        self.target_size = (int(args.image_size/2), int(args.image_size/2))
    def forward(self, x):
        x = self.channel_adjust(x)  # (B,64,7,7)
        x = nn.functional.interpolate(x, size = self.target_size, mode='bilinear', align_corners=False)
        return x

# class DeconvDVAE(nn.Module):
#     def __init__(self,args):
#         super().__init__()
#         # H2, W2 = (int(args.image_size/4), int(args.image_size/4))
#         self.layer1 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)
#         self.layer2 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)
#         self.layer3 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)

#     def forward(self,x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x

# class DeconvCNN(nn.Module):
#     def __init__(self,args):
#         super().__init__()
#         # H2, W2 = (int(args.image_size/4), int(args.image_size/4))
#         self.layer1 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)
#         self.layer2 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)
#         self.layer3 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)
#         self.layer4 = nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)

#     def forward(self,x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x

class ShuffleBlock(nn.Module):
    def __init__(self, in_ch=2048, bottleneck=512, r=2):
        super().__init__()
        self.pre   = nn.Conv2d(in_ch, bottleneck * (r ** 2), 1, bias=False)   # 2048,4,4 -> bottleneck*r*r,4,4
        self.shuffle = nn.PixelShuffle(r)                                     # bottleneck, 4*r,4*r
        self.post  = nn.Conv2d(bottleneck, in_ch, 1, bias=False)              # 2048, 4*r,4*r
        self.bn    = nn.BatchNorm2d(in_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pre(x)
        x = self.shuffle(x)
        x = self.post(x)
        return self.act(self.bn(x))

class UpsampleShuffle(nn.Module):
    def __init__(self, up_factor, channels=2048, bottleneck=512):
        super().__init__()
        n_stages = int(math.log2(up_factor))
        self.blocks = nn.ModuleList([
            ShuffleBlock(channels, bottleneck=bottleneck, r=2)
            for _ in range(n_stages)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class Deconv(nn.Module):
    def __init__(self,args, up_factor):
        super().__init__()
        self.up = UpsampleShuffle(channels=2048, up_factor = up_factor)

    def forward(self,x):
        return self.up(x)

class STEVE(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        
        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.img_channels = args.img_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model

        # dvae
        self.dvae = dVAE(args.vocab_size, args.img_channels)

        # encoder networks
        self.steve_encoder = STEVEEncoder(args)

        # decoder networks
        self.steve_decoder = STEVEDecoder(args)

        #backbone pretrained model(new class)
        self.backbone = CVCL_VISION_ENCODER()

        # deconv
        self.deconvDVAE = Deconv(args, up_factor = 8)
        self.deconvCNN = Deconv(args, up_factor = 16)


    def forward(self, video, tau, hard):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W
        print(video_flat.shape)
        cvcl_feats = self.backbone(video_flat)                            # B * T, 2048, 4, 4 
        print(cvcl_feats.shape)
        import sys
        sys.exit(0)

        # dvae encode
        cvcl_dvae = self.deconvDVAE(cvcl_feats)                                  # B * T, 2048, 32, 32
        # z_logits = F.log_softmax(self.dvae.encoder(video_flat), dim=1)       # B * T, vocab_size, H_enc, W_enc
        z_logits = F.log_softmax(cvcl_dvae, dim=1)                           # B * T, vocab_size, H_enc, W_enc
        # z_soft = gumbel_softmax(z_logits, tau, hard, dim=1)                  # B * T, vocab_size, H_enc, W_enc          soft one-hot codes 软离散码
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()         # B * T, vocab_size, H_enc, W_enc          hard one-hot codes 硬离散码
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                         # B * T, H_enc * W_enc, vocab_size
        z_emb = self.steve_decoder.dict(z_hard)                                                     # B * T, H_enc * W_enc, d_model
        z_emb = torch.cat([self.steve_decoder.bos.expand(B * T, -1, -1), z_emb], dim=1)             # B * T, 1 + H_enc * W_enc, d_model
        z_emb = self.steve_decoder.pos(z_emb)                                                       # B * T, 1 + H_enc * W_enc, d_model      code embeddings

        # dvae recon
        # dvae_recon = self.dvae.decoder(z_soft).reshape(B, T, C, H, W)               # B, T, C, H, W
        # dvae_mse = ((video - dvae_recon) ** 2).sum() / (B * T)                      # 1

        # savi
        cvcl_cnn = self.deconvCNN(cvcl_feats)         # B * T, 2048, 64, 64
        emb = self.steve_encoder.cnn(cvcl_cnn)      # B * T, d_model, H/2, W/2
        # emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        # emb = self.backbone(video_flat)
        # emb = self.transition(emb) 
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)         # slots: B, T, num_slots, slot_size
                                                                # attns: B, T, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)          # B, T, num_slots, 1, H, W
        attns = video.unsqueeze(2) * attns + (1. - attns)                               # B, T, num_slots, C, H, W

        # decode
        slots = self.steve_encoder.slot_proj(slots)                                                         # B, T, num_slots, d_model
        pred = self.steve_decoder.tf(z_emb[:, :-1], slots.flatten(end_dim=1))                               # B * T, H_enc * W_enc, d_model
        pred = self.steve_decoder.head(pred)                                                                # B * T, H_enc * W_enc, vocab_size
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / (B * T)                         # 1

        # return (dvae_recon.clamp(0., 1.),
        #         cross_entropy,
        #         dvae_mse,
        #         attns)
        return (None, cross_entropy, None, attns)

    def encode(self, video):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)

        # savi
        emb = self.steve_encoder.cnn(video_flat)      # B * T, cnn_hidden_size, H, W
        emb = self.steve_encoder.pos(emb)             # B * T, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.steve_encoder.mlp(self.steve_encoder.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.d_model)                                                # B, T, H * W, cnn_hidden_size

        slots, attns = self.steve_encoder.savi(emb_set)     # slots: B, T, num_slots, slot_size
                                                            # attns: B, T, num_slots, num_inputs

        attns = attns \
            .transpose(-1, -2) \
            .reshape(B, T, self.num_slots, 1, H_enc, W_enc) \
            .repeat_interleave(H // H_enc, dim=-2) \
            .repeat_interleave(W // W_enc, dim=-1)                      # B, T, num_slots, 1, H, W

        attns_vis = video.unsqueeze(2) * attns + (1. - attns)           # B, T, num_slots, C, H, W

        return slots, attns_vis, attns

    def decode(self, slots):
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        slots = self.steve_encoder.slot_proj(slots)

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.steve_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.steve_decoder.tf(
                self.steve_decoder.pos(input),
                slots
            )
            z_next = F.one_hot(self.steve_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.steve_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, video):
        """
        image: batch_size x img_channels x H x W
        """
        B, T, C, H, W = video.size()
        slots, attns, _ = self.encode(video)
        recon_transformer = self.decode(slots.flatten(end_dim=1))
        recon_transformer = recon_transformer.reshape(B, T, C, H, W)

        return recon_transformer
    
    def visual_cvcl(self, video):
        B, T, C, H, W = video.size()

        video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W
        feats = self.backbone(video_flat)                            # B * T, 2048, 32, 32
        orig_img_np =  video_flat[0].permute(1, 2, 0).cpu().numpy()  # shape = [H, W, C]
        import numpy as np
        from scipy.spatial.distance import cdist
        import matplotlib.pyplot as plt
        from PIL import Image
        import os

        feat = feats[0]
        D, H, W = feat.shape
        feats = feat.view(D, -1).permute(1,0).cpu().numpy()  # shape = [1024,2048]

        def kmeans_plus_plus(X, K):
            # X: (N,D)
            N, _ = X.shape
            centers = []
            # 随机选一个初始中心
            idx = np.random.choice(N)
            centers.append(X[idx])
            for _ in range(1, K):
                # 计算每个点到已有 centers 的最小距离平方
                d2 = np.min(cdist(X, np.stack(centers)), axis=1)
                probs = d2 / d2.sum()
                idx = np.random.choice(N, p=probs)
                centers.append(X[idx])
            return np.stack(centers)  # (K,D)
        
        K = 10
        centers = kmeans_plus_plus(feats, K)
        dists = cdist(feats, centers)  # shape = [1024, K]
        labels = np.argmin(dists, axis=1)  # shape = [1024,]
        label_map = labels.reshape(H, W)

        label_up = np.repeat(np.repeat(label_map, 4, axis=0), 4, axis=1)

        # 2. 构造彩色覆盖图
        cmap = plt.get_cmap('tab20', K)
        colors = cmap(label_up)            # RGBA, shape = [128,128,4]
        colors_rgb = (colors[...,:3] * 255).astype(np.uint8)

        # 3. 原图转 uint8
        orig_uint8 = (orig_img_np * 255).astype(np.uint8)

        # 4. 用 PIL 叠加
        orig_img_pil = Image.fromarray(orig_uint8)
        overlay_pil  = Image.blend(orig_img_pil, Image.fromarray(colors_rgb), alpha=0.5)

        # 5. 保存
        save_path = "/mnt/data/overlay_cluster_sample0.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        overlay_pil.save(save_path)

        print(f"已将叠加可视化结果保存到：{save_path}")

        # plt.figure(figsize=(4,4))
        # plt.imshow(label_map, cmap='tab20', interpolation='nearest')
        # plt.axis('off')
        # plt.title(f"Sample 0, K={K} 聚类结果")
        # plt.savefig("/home/wz3008/baby_objectcentric/image/", dpi=300, bbox_inches='tight')
        # plt.close() 

