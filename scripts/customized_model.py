import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import wandb
import random
from sklearn.metrics import roc_auc_score, recall_score
import argparse
import torch.nn.functional as F
import uuid
from torchsummary import summary

class DataGenerator(Dataset):
    def __init__(self, directory, data_augmentation=False, phase='train'):
        self.phase=phase
        self.directory = directory
        self.data_aug = data_augmentation
        self.X_path, self.Y_dict = self.search_data()
        self.print_stats()

    def __len__(self):
        return len(self.X_path)

    def __getitem__(self, index):
        data, label = self.data_generation(self.X_path[index])
        return data.float(), label

    def load_data(self, path):
        data = np.load(path, mmap_mode='r', allow_pickle= True)
        data = self.uniform_sampling(data, target_frames=64)
        if self.data_aug:
            data[..., :3] = self.color_jitter(data[..., :3]) 
            data = self.random_flip(data, prob=0.5)
        data[..., :3] = self.normalize(data[..., :3])
        data[..., 3:] = self.normalize(data[..., 3:])
        return data

    def normalize(self, data):
        mean = data.mean()
        std = data.std()
        return (data - mean) / std

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(video, (2,))
        return video

    def uniform_sampling(self, video, target_frames=64):
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])
        num_pad = target_frames - len(sampled_video)
        if num_pad>0:
            for i in range(-num_pad,0):
                try:
                    sampled_video.append(video[i])
                except:
                    sampled_video.append(video[0])
        return np.array(sampled_video, dtype=np.float32)

    def color_jitter(self, video):
        s_jitter = np.random.uniform(-0.2, 0.2) 
        v_jitter = np.random.uniform(-30, 30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(np.array(video[i]), cv2.COLOR_RGB2HSV)
            s = hsv[..., 1] + s_jitter
            v = hsv[..., 2] + v_jitter
            s[s < 0] = 0
            s[s > 1] = 1
            v[v < 0] = 0
            v[v > 255] = 255
            hsv[..., 1] = s
            hsv[..., 2] = v
            video[i] = torch.Tensor(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        return video

    def print_stats(self):
        self.dirs = sorted(os.listdir(self.directory))
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        print("Found {} files belonging to {} classes.".format(self.n_files, self.n_classes))
        for i, label in enumerate(self.dirs):
            print('{:10s} : {}'.format(label, i))

    def search_data(self):
        X_path = []
        Y_dict = {}
        self.dirs = sorted(os.listdir(self.directory))
        label_dict = {folder: i for i, folder in enumerate(self.dirs)}

        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            for file in os.listdir(folder_path):
                path = os.path.join(folder_path,file)
                X_path.append(path)
                Y_dict[path] = label_dict[folder]               
        return X_path, Y_dict

    def data_generation(self, batch_path):
        batch_x = self.load_data(batch_path)
        batch_y = self.Y_dict[batch_path] 
        batch_x = torch.tensor(batch_x.copy())
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        return batch_x, batch_y

class MCA(nn.Module):
    def __init__(self, dim=64, num_heads=8, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self.attention_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)

    def forward(self, x_q, x_k, x_v):
        B, dim_Q, C = x_q.shape
        _, dim_KV, _ = x_k.shape

        query = self.q(x_q).reshape(B, dim_Q, self.num_heads, C//self.num_heads).permute(0,2,1,3).contiguous()
        key = self.k(x_k).reshape(B, dim_KV, self.num_heads, C//self.num_heads).permute(0,2,1,3).contiguous()
        value = self.v(x_v).reshape(B, dim_KV, self.num_heads, C//self.num_heads).permute(0,2,1,3).contiguous()

        attention = (query @ key.transpose(-2,-1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_drop(attention)

        x = (attention @ value).transpose(-2,-1).reshape(B,dim_Q,C)
        x=self.proj(x)
        return x

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.relu=nn.ReLU(inplace=True)

        self.rgbblock1 = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )
        self.rgbblock2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )
        self.rgbblock3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )
        self.rgbblock4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )

        self.optblock1 = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )
        self.optblock2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )
        self.optblock3 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )
        self.optblock4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2))
        )

        self.embedding_dim = 64
        self.embedding_layer = nn.Linear(32*14*14, self.embedding_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embedding_dim))

        self.mca = MCA(dim=self.embedding_dim)

        self.fc1 = nn.Linear(self.embedding_dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

        self.__init_weight()

    def forward(self, x):
        rgb = x[...,:3]
        opt = x[...,3:5]
        rgb = rgb.contiguous().view(-1, 3, 64, rgb.shape[2], rgb.shape[3])
        opt = opt.contiguous().view(-1, 2, 64, opt.shape[2], opt.shape[3])

        rgb = self.rgbblock1(rgb)
        rgb = self.rgbblock2(rgb)
        rgb = self.rgbblock3(rgb)
        rgb = self.rgbblock4(rgb)

        opt = self.optblock1(opt)
        opt = self.optblock2(opt)
        opt = self.optblock3(opt)
        opt = self.optblock4(opt)

        rgb = rgb.permute(0,2,1,3,4).contiguous()
        opt = opt.permute(0,2,1,3,4).contiguous()
        B, T, C, H, W = rgb.shape

        rgb_seq = rgb.view(B, T, C*H*W)
        opt_seq = opt.view(B, T, C*H*W)
        rgb_seq = self.embedding_layer(rgb_seq)
        opt_seq = self.embedding_layer(opt_seq)

        fused = self.mca(rgb_seq, opt_seq, opt_seq)  # (B, T, D)

        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        fused_with_cls = torch.cat([cls_token, fused], dim=1) # (B, T+1, D)
        fused_final = self.mca(fused_with_cls, fused_with_cls, fused_with_cls)  # (B, T+1, D)

        x_cls = fused_final[:, 0, :]  # (B, D)

        x = self.fc1(x_cls)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

if __name__ == "__main__":
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=100) 
    parer.add_argument('--weight_decay', type=float, default=1e-5)
    parer.add_argument('--batch_size', type=int, default=1)
    parer.add_argument('--lr', type=float, default=1e-4) 
    parer.add_argument('--step_size', type=int, default=100)
    parer.add_argument('--momentum',type=float, default=0.9)
    parer.add_argument('--root', type=str, default='./RWF2000')
    parer.add_argument('--log_dir', type=str, default='./log')
    parer.add_argument('--name', type=str, default='RWF2000')
    ops = parer.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FusionModel().to(device)
    optimizer = optim.AdamW(model.parameters(),
                                    lr=ops.lr,
                                    weight_decay=ops.weight_decay)
    criterion = nn.CrossEntropyLoss()

    trainset_path = '/home/jeong/Desktop/my_work/Class/introductiontocv/CV_project/npy_dataset/train'
    validation_path = '/home/jeong/Desktop/my_work/Class/introductiontocv/CV_project/npy_dataset/val'

    train_dataset = DataGenerator(directory=trainset_path, data_augmentation=True)
    train_loader = DataLoader(train_dataset, batch_size = ops.batch_size, shuffle=True, num_workers=0)

    val_dataset = DataGenerator(directory=validation_path, phase='val')
    val_loader = DataLoader(val_dataset, batch_size = ops.batch_size, num_workers=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    def _train(epoch, device):
        model.train()
        train_correct = 0
        train_total = 0
        running_train_loss = []

        for batch_idx, (video, target) in enumerate(train_loader):
            video, target = video.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(video)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_train_loss.append(loss.item())
            _, pred = torch.max(output, 1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if batch_idx % ops.step_size == 0:
                wandb.log({
                    'epoch': epoch,
                    'step': batch_idx + epoch * len(train_loader),
                    'batch_train_loss': loss.item(),
                    'lr': lr,
                })

        train_loss = np.mean(running_train_loss)
        train_acc = train_correct / train_total
        return train_acc, train_loss

    def _val(model, device):
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_target = []
        all_prob = []

        with torch.no_grad():
            for video, target in val_loader:
                video, target = video.to(device), target.to(device)
                output = model(video)
                loss = criterion(output, target)
                val_loss += loss.item()

                _, pred = torch.max(output, 1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)

                probs = F.softmax(output, dim=1)[:,0].cpu().numpy()
                targets = target.cpu().numpy()
                all_prob.extend(probs)
                all_target.extend(targets)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        roc_auc = roc_auc_score(all_target, all_prob)
        pred_labels = (np.array(all_prob) >= 0.5).astype(int)
        recall = recall_score(all_target, pred_labels)
        return val_loss, val_accuracy, roc_auc, recall

    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device =='cuda':
        print('gpu device is using')
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    wandb.login()
    wandb.init(
        project='computer vision',
        name=f"{ops.name}-{uuid.uuid4().hex[:8]}",
        settings= wandb.Settings(code_dir="."),
        save_code=True
    )

    for epoch in tqdm(range(1, ops.epoch + 1), desc="Epoch Progress"):
        train_acc, train_loss = _train(epoch, device)
        val_loss, val_accuracy, roc_auc, recall = _val(model, device)

        wandb.log({
            'epoch': epoch,
            'val_loss': val_loss,
            'val_acc': val_accuracy,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'recall' : recall,
        })
        scheduler.step(val_loss)
