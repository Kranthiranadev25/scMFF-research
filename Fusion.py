import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def pad_features(*features, pad_value=0.0):
    max_length = max(len(f) for f in features)
    padded_features = [np.pad(f, (0, max_length - len(f)), constant_values=pad_value) for f in features]
    return np.array(padded_features)
def weighted_sum_fusion(features, weights):
    features = pad_features(*features) 
    weights = np.array(weights) / np.sum(weights) 
    return np.sum(features.T * weights, axis=1)
def hadamard_product_fusion(features):
    features = pad_features(*features)
    fused_feature = np.prod(features, axis=0)
    return fused_feature

def pad_features_tensor(features, pad_value=0.0):
    max_length = max(len(f) for f in features)
    padded_features = [torch.cat([f, torch.full((max_length - len(f),), pad_value)]) for f in features]
    return torch.stack(padded_features)
class AttentionFusionClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(AttentionFusionClassifier, self).__init__()
        self.W = nn.Parameter(torch.randn(feature_dim)) 
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes) 
        )
    def forward(self, features):
        features = pad_features_tensor(features)  
        scores = torch.matmul(features, self.W) 
        attention_weights = torch.softmax(scores, dim=0) 
        fused_feature = torch.sum(features.T * attention_weights, dim=1) 
        logits = self.classifier(fused_feature) 
        return fused_feature, logits
def train_AttentionFusion(features, labels, num_classes, num_epochs=100, lr=0.01):
    feature_dim = max(len(f) for f in features)  
    model = AttentionFusionClassifier(feature_dim, num_classes)  
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    criterion = nn.CrossEntropyLoss()  
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        fused_feature, logits = model(features) 
        loss = criterion(logits, labels) 
        loss.backward() 
        optimizer.step()  
    return model



def zero_pad_feature(feature, max_dim):
    feature = np.array(feature) 
    padded_feature = np.pad(feature, (0, max_dim - len(feature)), mode='constant', constant_values=0)
    return torch.tensor(padded_feature, dtype=torch.float32).unsqueeze(0)

class MoEFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4):
        super(MoEFusion, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
    def forward(self, *features):
        x = torch.cat(features, dim=-1)
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        fused_output = torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=-1)
        return fused_output

class ResidualFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualFusion, self).__init__()
        self.linear = nn.Linear(input_dim * 4, output_dim)
        self.residual = nn.Linear(input_dim * 4, output_dim)
    def forward(self, *features):
        x = torch.cat(features, dim=-1)
        return F.relu(self.linear(x) + self.residual(x))

class TransformerConcatFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, num_layers=2):
        super(TransformerConcatFusion, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(output_dim * 4, output_dim)
    def forward(self, *features):
        x = torch.stack([self.embedding(f) for f in features], dim=0)
        x = self.transformer(x).view(x.shape[1], -1) 
        return self.fc(x)