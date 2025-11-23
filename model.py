import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

'''
    Robust Sensor-Based Activity Recognition via Physics-Aware Disentangled Representation Learning and FiLM Conditioning
  
    Gyuyeon Lim and Myung-Kyu Yi
'''

def extract_physics_features(x):
    B, C, T = x.shape
    features = []
    if C >= 3:
        acc = x[:, :3, :]
        gravity_dir = acc.mean(dim=-1)
        gravity_dir = F.normalize(gravity_dir, dim=1, eps=1e-8)
        features.append(gravity_dir)
    if C >= 3:
        acc = x[:, :3, :]
        jerk = torch.diff(acc, dim=-1)
        jerk_mag = jerk.norm(dim=1).mean(dim=-1, keepdim=True)
        features.append(jerk_mag)
    energy = (x ** 2).mean(dim=(1, 2)).unsqueeze(1)
    features.append(energy)
    x_fft = torch.fft.rfft(x, dim=-1)
    power = x_fft.abs() ** 2
    freq_bins = torch.arange(power.shape[-1], device=x.device).float()
    power_mean = power.mean(dim=1)
    freq_center = (power_mean * freq_bins).sum(dim=-1) / (power_mean.sum(dim=-1) + 1e-8)
    freq_center = freq_center.unsqueeze(1)
    features.append(freq_center)
    psi = torch.cat(features, dim=-1)
    return psi

class DoAugmentation:
    @staticmethod
    def apply_random_augmentation(x):
        x = x.clone()
        if torch.rand(1) < 0.5:
            angle = torch.rand(1) * 2 * 3.14159
            rot_matrix = torch.tensor([
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle), 0],
                [0, 0, 1]
            ], device=x.device, dtype=x.dtype)
            if x.shape[1] >= 3:
                acc = x[:, :3, :]
                acc_rot = torch.einsum('ij,bjt->bit', rot_matrix, acc)
                x = torch.cat([acc_rot, x[:, 3:, :]], dim=1)
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        if torch.rand(1) < 0.3:
            factor = torch.rand(1) * 0.4 + 0.8
            try:
                B, C, T = x.shape
                new_length = max(1, int(T * factor))
                x_reshaped = x.view(B * C, 1, T)
                x_resized = F.interpolate(x_reshaped, size=new_length, mode='linear', align_corners=False)
                x_resized = F.interpolate(x_resized, size=T, mode='linear', align_corners=False)
                x = x_resized.view(B, C, T)
            except:
                pass
        return x

class CompactEncoder(nn.Module):
    def __init__(self, in_channels, stage1_channels, stage2_channels, stage3_channels, 
                 kernel_size, stride1, stride2, stride3):
        super().__init__()
        self.stage1 = self._make_stage(in_channels, stage1_channels, stride=stride1, kernel_size=kernel_size)
        self.stage2 = self._make_stage(stage1_channels, stage2_channels, stride=stride2, kernel_size=kernel_size)
        self.stage3 = self._make_stage(stage2_channels, stage3_channels, stride=stride3, kernel_size=kernel_size)

    def _make_stage(self, in_ch, out_ch, stride=1, kernel_size=3):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class CompactDecoder(nn.Module):
    def __init__(self, stage3_channels, stage2_channels, stage1_channels, out_channels,
                 kernel_size, upsample_kernel, output_padding1, output_padding2, output_padding3,
                 target_length):
        super().__init__()
        self.target_length = target_length
        self.stage1 = self._make_upsample(stage3_channels, stage2_channels, 
                                          upsample_kernel, output_padding1, kernel_size)
        self.stage2 = self._make_upsample(stage2_channels, stage1_channels, 
                                          upsample_kernel, output_padding2, kernel_size)
        self.stage3 = self._make_upsample(stage1_channels, out_channels, 
                                          upsample_kernel, output_padding3, kernel_size)
        self.adjust_length = nn.Conv1d(out_channels, out_channels, 
                                       kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def _make_upsample(self, in_ch, out_ch, upsample_kernel, output_padding, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=upsample_kernel, stride=2, 
                             padding=1, output_padding=output_padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

    def forward(self, h):
        x = self.stage1(h)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.adjust_length(x)
        if x.size(-1) > self.target_length:
            x = x[:, :, :self.target_length]
        elif x.size(-1) < self.target_length:
            x = F.pad(x, (0, self.target_length - x.size(-1)))
        return x

class CausalManifoldHAR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_film = args.use_film
        self.use_decoder = args.use_decoder
        
        self.encoder = CompactEncoder(
            in_channels=args.in_channels,
            stage1_channels=args.stage1_channels,
            stage2_channels=args.stage2_channels,
            stage3_channels=args.stage3_channels,
            kernel_size=args.kernel_size,
            stride1=args.stride1,
            stride2=args.stride2,
            stride3=args.stride3
        )
        
        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(args.physics_features, args.film_hidden),
                nn.GELU(),
                nn.Linear(args.film_hidden, args.stage3_channels * 2)
            )
        else:
            self.film = None
        
        self.to_Z = nn.Conv1d(args.stage3_channels, args.latent_channels, 1)
        self.to_N = nn.Conv1d(args.stage3_channels, args.latent_channels, 1)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(args.latent_channels, args.cls_hidden1),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.cls_hidden1, args.cls_hidden2),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.cls_hidden2, args.num_classes)
        )
        
        if self.use_decoder:
            self.decoder = CompactDecoder(
                stage3_channels=args.latent_channels,
                stage2_channels=args.stage2_channels,
                stage1_channels=args.stage1_channels,
                out_channels=args.in_channels,
                kernel_size=args.kernel_size,
                upsample_kernel=args.upsample_kernel,
                output_padding1=args.output_padding1,
                output_padding2=args.output_padding2,
                output_padding3=args.output_padding3,
                target_length=args.window_size
            )
        else:
            self.decoder = None

    def forward(self, x, psi=None):
        h = self.encoder(x)
        if self.use_film and self.film is not None and psi is not None:
            params = self.film(psi).view(-1, 2, h.size(1), 1)
            gamma, beta = params[:, 0], params[:, 1]
            h = h * gamma + beta
        h_z = self.to_Z(h)
        h_n = self.to_N(h)
        logits = self.classifier(h_z)
        x_recon = self.decoder(h_z + h_n) if self.use_decoder and self.decoder is not None else None
        return logits, h_z, h_n, x_recon

    def extract_features(self, x):
        h = self.encoder(x)
        if self.use_film and self.film is not None:
            psi = extract_physics_features(x)  
            params = self.film(psi).view(-1, 2, h.size(1), 1)
            gamma, beta = params[:, 0], params[:, 1]
            h = h * gamma + beta
        return self.to_Z(h)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--stage1_channels', type=int, default=16)
    parser.add_argument('--stage2_channels', type=int, default=32)
    parser.add_argument('--stage3_channels', type=int, default=64)
    parser.add_argument('--latent_channels', type=int, default=64)
    
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride1', type=int, default=2)
    parser.add_argument('--stride2', type=int, default=2)
    parser.add_argument('--stride3', type=int, default=2)
    
    parser.add_argument('--upsample_kernel', type=int, default=4)
    parser.add_argument('--output_padding1', type=int, default=0)
    parser.add_argument('--output_padding2', type=int, default=1)
    parser.add_argument('--output_padding3', type=int, default=0)
    
    parser.add_argument('--physics_features', type=int, default=6)
    parser.add_argument('--film_hidden', type=int, default=32)
    
    parser.add_argument('--cls_hidden1', type=int, default=128)
    parser.add_argument('--cls_hidden2', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--window_size', type=int, default=40)
    
    parser.add_argument('--use_film', type=int, default=1)
    parser.add_argument('--use_decoder', type=int, default=1)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    args.use_film = bool(args.use_film)
    args.use_decoder = bool(args.use_decoder)
    model = CausalManifoldHAR(args)
    print(model)
