import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SkinDiseaseCNN, self).__init__()
        
        # Feature Extraction - Using residual connections and deeper architecture
        # First block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        
        # Second block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        
        # Third block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        
        # Fourth block
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        
        # Pooling and regularization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)  # Reduced dropout to 0.3 from 0.5
        self.dropout2d = nn.Dropout2d(0.2)  # Adding spatial dropout
        
        # Attention mechanism
        self.attention = SelfAttention(256)
        
        # Adaptive Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.global_max_pool = nn.AdaptiveMaxPool2d((4, 4))
        
        # Fully Connected Layers with batch normalization
        self.fc1 = nn.Linear(256 * 4 * 4 * 2, 512)  # *2 for concatenated avg and max pool
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1 with residual connection
        residual = F.interpolate(x, scale_factor=1) if x.size(1) == 64 else self.conv1(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        x = x + residual if x.size() == residual.size() else x
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Block 2 with residual connection
        residual = F.interpolate(x, scale_factor=1) if x.size(1) == 128 else self.conv2(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn2_1(self.conv2_1(x)))
        x = x + residual if x.size() == residual.size() else x
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Block 3 with residual connection
        residual = F.interpolate(x, scale_factor=1) if x.size(1) == 256 else self.conv3(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn3_1(self.conv3_1(x)))
        x = x + residual if x.size() == residual.size() else x
        x = self.pool(x)
        x = self.dropout2d(x)
        
        # Block 4 with residual connection
        residual = x
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn4_1(self.conv4_1(x)))
        x = x + residual
        x = self.pool(x)
        
        # Apply attention mechanism
        x = self.attention(x)
        
        # Multi-level pooling (both average and max)
        avg_pooled = self.global_avg_pool(x)
        max_pooled = self.global_max_pool(x)
        
        # Concatenate different pooling results
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected with batch normalization
        x = F.leaky_relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Final classification layer
        
        return x


# Self-Attention module to focus on important features
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=2)
        
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        out = self.gamma * out + x
        return out