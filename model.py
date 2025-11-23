import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
import numpy as np

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
    def __init__(self, rotation_prob=0.5, noise_prob=0.5, noise_std=0.02, 
                 warp_prob=0.3, warp_factor_min=0.8, warp_factor_max=1.2):
        self.rotation_prob = rotation_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.warp_prob = warp_prob
        self.warp_factor_min = warp_factor_min
        self.warp_factor_max = warp_factor_max
    
    def apply_random_augmentation(self, x):
        x = x.clone()
        
        if torch.rand(1) < self.rotation_prob:
            x = self._apply_so3_rotation(x)
        
        if torch.rand(1) < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        if torch.rand(1) < self.warp_prob:
            x = self._apply_temporal_warping(x)
        
        return x
    
    def _apply_so3_rotation(self, x):
        device = x.device
        dtype = x.dtype
        
        theta = torch.rand(1, device=device, dtype=dtype) * 2 * 3.14159
        axis = torch.rand(3, device=device, dtype=dtype)
        axis = F.normalize(axis, dim=0, eps=1e-8)
        
        K = torch.zeros((3, 3), device=device, dtype=dtype)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + sin_theta * K + (1 - cos_theta) * torch.mm(K, K)
        
        x_aug_list = []
        if x.shape[1] >= 3:
            x_aug_list.append(torch.einsum('ij,bjt->bit', R, x[:, 0:3, :]))
        if x.shape[1] >= 6:
            x_aug_list.append(torch.einsum('ij,bjt->bit', R, x[:, 3:6, :]))
        if x.shape[1] >= 9:
            x_aug_list.append(torch.einsum('ij,bjt->bit', R, x[:, 6:9, :]))
        
        if len(x_aug_list) == 3:
            return torch.cat(x_aug_list, dim=1)
        elif len(x_aug_list) > 0:
            rotated = torch.cat(x_aug_list, dim=1)
            remaining_channels = x.shape[1] - rotated.shape[1]
            if remaining_channels > 0:
                return torch.cat([rotated, x[:, -remaining_channels:, :]], dim=1)
            return rotated
        else:
            return x
    
    def _apply_temporal_warping(self, x):
        try:
            B, C, T = x.shape
            factor_range = self.warp_factor_max - self.warp_factor_min
            factor = torch.rand(1).item() * factor_range + self.warp_factor_min
            new_length = max(1, int(T * factor))
            
            x_reshaped = x.view(B * C, 1, T)
            x_resized = F.interpolate(x_reshaped, size=new_length, mode='linear', align_corners=False)
            x_resized = F.interpolate(x_resized, size=T, mode='linear', align_corners=False)
            
            return x_resized.view(B, C, T)
        except:
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

class CausalHARLoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_inv=0.1, lambda_dis=0.01,
                 use_reconstruction=True, use_invariance=True, use_disentanglement=True,
                 stft_n_fft=32, stft_hop_length=None):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_inv = lambda_inv
        self.lambda_dis = lambda_dis
        self.use_reconstruction = use_reconstruction
        self.use_invariance = use_invariance
        self.use_disentanglement = use_disentanglement
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length if stft_hop_length else stft_n_fft // 2
        self.ce = nn.CrossEntropyLoss()

    def stft_loss(self, x, x_recon):
        try:
            B, C, T = x.shape
            x_flat = x.view(-1, T)
            x_recon_flat = x_recon.view(-1, T)
            n_fft = min(self.stft_n_fft, T // 2)
            hop_length = n_fft // 2
            win_length = n_fft
            if n_fft < 4:
                return torch.tensor(0.0, device=x.device)
            window = torch.hann_window(win_length, device=x.device)
            x_stft = torch.stft(x_flat, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length, window=window,
                               return_complex=True, center=False)
            x_recon_stft = torch.stft(x_recon_flat, n_fft=n_fft, hop_length=hop_length,
                                     win_length=win_length, window=window,
                                     return_complex=True, center=False)
            return F.mse_loss(x_stft.abs(), x_recon_stft.abs())
        except:
            return torch.tensor(0.0, device=x.device)

    def hsic_loss(self, z, n):
        try:
            batch_size = z.size(0)
            z = z.view(batch_size, -1)
            n = n.view(batch_size, -1)
            z_norm = F.normalize(z, dim=1, eps=1e-8)
            n_norm = F.normalize(n, dim=1, eps=1e-8)
            Kz = torch.mm(z_norm, z_norm.t())
            Kn = torch.mm(n_norm, n_norm.t())
            H = torch.eye(batch_size, device=z.device) - 1.0 / batch_size
            hsic_value = torch.trace(torch.mm(torch.mm(Kz, H), torch.mm(Kn, H))) / ((batch_size - 1) ** 2)
            return hsic_value
        except:
            return torch.tensor(0.0, device=z.device)

    def forward(self, logits, y, x, x_recon=None, z=None, n=None, z_prime=None):
        ce_loss = self.ce(logits, y)
        
        rec_loss = torch.tensor(0.0, device=logits.device)
        if self.use_reconstruction and x_recon is not None:
            rec_loss = 0.5 * F.mse_loss(x_recon, x) + 0.5 * self.stft_loss(x, x_recon)
        
        inv_loss = torch.tensor(0.0, device=logits.device)
        if self.use_invariance and z is not None and z_prime is not None:
            inv_loss = F.mse_loss(z, z_prime)
        
        dis_loss = torch.tensor(0.0, device=logits.device)
        if self.use_disentanglement and z is not None and n is not None:
            dis_loss = self.hsic_loss(z, n)
        
        total_loss = ce_loss + self.lambda_rec * rec_loss + self.lambda_inv * inv_loss + self.lambda_dis * dis_loss
        
        return total_loss, ce_loss.item(), rec_loss.item(), inv_loss.item(), dis_loss.item()

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, augmentation, device, args):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.augmentation = augmentation
        self.device = device
        self.args = args
        self.best_loss = float('inf')
        self.best_epoch = 0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_rec = 0
        total_inv = 0
        total_dis = 0
        num_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            psi = extract_physics_features(x)
            x_aug = self.augmentation.apply_random_augmentation(x)
            psi_aug = extract_physics_features(x_aug)
            
            logits, z, n, x_recon = self.model(x, psi)
            _, z_prime, _, _ = self.model(x_aug, psi_aug)
            
            loss, ce, rec, inv, dis = self.loss_fn(logits, y, x, x_recon, z, n, z_prime)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_ce += ce
            total_rec += rec
            total_inv += inv
            total_dis += dis
            num_batches += 1
        
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_ce = total_ce / max(num_batches, 1)
        avg_rec = total_rec / max(num_batches, 1)
        avg_inv = total_inv / max(num_batches, 1)
        avg_dis = total_dis / max(num_batches, 1)
        
        if epoch % self.args.print_freq == 0:
            print(f"Epoch {epoch}/{self.args.epochs} | Loss: {avg_loss:.6f} | CE: {avg_ce:.6f} | "
                  f"Rec: {avg_rec:.6f} | Inv: {avg_inv:.6f} | Dis: {avg_dis:.6f}")
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_epoch = epoch
            if self.args.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(self.args.save_dir, 'best_model.pt'))
        
        return avg_loss, avg_ce, avg_rec, avg_inv, avg_dis

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                psi = extract_physics_features(x)
                logits, z, n, x_recon = self.model(x, psi)
                loss, _, _, _, _ = self.loss_fn(logits, y, x, x_recon, z, n)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

def get_args():
    parser = argparse.ArgumentParser(description='CausalManifoldHAR Training')
    
    parser.add_argument('--in_channels', type=int, default=9)
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
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--window_size', type=int, default=128)
    
    parser.add_argument('--use_film', type=int, default=1)
    parser.add_argument('--use_decoder', type=int, default=1)
    
    parser.add_argument('--rotation_prob', type=float, default=0.5)
    parser.add_argument('--noise_prob', type=float, default=0.5)
    parser.add_argument('--noise_std', type=float, default=0.02)
    parser.add_argument('--warp_prob', type=float, default=0.3)
    parser.add_argument('--warp_factor_min', type=float, default=0.8)
    parser.add_argument('--warp_factor_max', type=float, default=1.2)
    
    parser.add_argument('--lambda_rec', type=float, default=1.0)
    parser.add_argument('--lambda_inv', type=float, default=0.1)
    parser.add_argument('--lambda_dis', type=float, default=0.01)
    parser.add_argument('--use_reconstruction', type=int, default=1)
    parser.add_argument('--use_invariance', type=int, default=1)
    parser.add_argument('--use_disentanglement', type=int, default=1)
    parser.add_argument('--stft_n_fft', type=int, default=32)
    parser.add_argument('--stft_hop_length', type=int, default=None)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--print_freq', type=int, default=25)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    args = get_args()
    
    args.use_film = bool(args.use_film)
    args.use_decoder = bool(args.use_decoder)
    args.use_reconstruction = bool(args.use_reconstruction)
    args.use_invariance = bool(args.use_invariance)
    args.use_disentanglement = bool(args.use_disentanglement)
    args.save_model = bool(args.save_model)
    
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
    
    model = CausalManifoldHAR(args)
    
    augmentation = DoAugmentation(
        rotation_prob=args.rotation_prob,
        noise_prob=args.noise_prob,
        noise_std=args.noise_std,
        warp_prob=args.warp_prob,
        warp_factor_min=args.warp_factor_min,
        warp_factor_max=args.warp_factor_max
    )
    
    loss_fn = CausalHARLoss(
        lambda_rec=args.lambda_rec,
        lambda_inv=args.lambda_inv,
        lambda_dis=args.lambda_dis,
        use_reconstruction=args.use_reconstruction,
        use_invariance=args.use_invariance,
        use_disentanglement=args.use_disentanglement,
        stft_n_fft=args.stft_n_fft,
        stft_hop_length=args.stft_hop_length
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    trainer = Trainer(model, loss_fn, optimizer, scheduler, augmentation, device, args)
