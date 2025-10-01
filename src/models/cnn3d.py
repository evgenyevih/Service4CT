from __future__ import annotations
from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D(nn.Module):
    """3D CNN с архитектурой для медицинских изображений"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(CNN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Классификатор
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate * 0.6)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 1, depth, height, width]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Классификатор
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return torch.sigmoid(x)


class CNN3DModel:
    """3D CNN classifier wrapper with numpy I/O using CNN3D."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu", depth_size: int = 64, spatial_size: int = 64):
        self.device = torch.device(device)
        self.net = CNN3D(num_classes=2, dropout_rate=0.3).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                self.net.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"⚠️ Ошибка загрузки весов: {e}")
                pass
        self.net.eval()
        self.depth_size = int(depth_size)
        self.spatial_size = int(spatial_size)

    def predict_probability(self, volume: np.ndarray) -> float:
        if volume.size == 0:
            return 0.5
        # Resize to expected 3D input size
        import torch.nn.functional as F
        vol = torch.from_numpy(volume).float()  # [D,H,W]
        vol = vol.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        
        # Ensure input is NDH W => (N,C,D,H,W) and output is (D,H,W)
        vol = F.interpolate(vol, size=(self.depth_size, self.spatial_size, self.spatial_size), mode="trilinear", align_corners=False)
        x = vol.to(self.device)
        with torch.no_grad():
            # Получаем предсказания для обоих классов и берем вероятность патологии (индекс 1)
            predictions = self.net(x)
            p = predictions[0, 1].item()  # Берем вероятность патологии
        return float(np.clip(p, 0.0, 1.0))


