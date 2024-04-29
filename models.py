import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import random
import timm
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
# !pip install torch efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import BertForSequenceClassification, XLNetForSequenceClassification

import warnings
warnings.filterwarnings('ignore')

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.image_model = EfficientNet.from_pretrained('efficientnet-b3')
        torch.nn.Sequential(*list(self.image_model.children())[:-1])
        num_features = self.image_model._fc.in_features
        self.image_model._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        # Text Model (BERT)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_linear = nn.Linear(self.text_model.config.hidden_size, 512)
        # Fusion Layer
        self.fusion_linear = nn.Linear(1024, num_classes)
    def forward(self, image_input, text_input_ids, text_attention_mask):
        # Process image
        image_features = self.image_model(image_input)
        # Process text
        text_outputs = self.text_model(text_input_ids, attention_mask=text_attention_mask)
        pooled_text_output = text_outputs.pooler_output
        text_features = self.text_linear(pooled_text_output)
        # Concatenate image and text features
        multimodal_features = torch.cat((image_features, text_features), dim=1)
        # Fusion layer
        combined_logits = self.fusion_linear(multimodal_features)
        return combined_logits



class AttentionLayer(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, attention_dim)
        self.value_fc = nn.Linear(feature_dim, feature_dim)
        self.query = nn.Parameter(torch.randn(attention_dim), requires_grad=True)

    def forward(self, features):
        # Compute attention scores
        attention_scores = F.gelu(self.attention_fc(features))  # GELU for nonlinear activation
        attention_scores = torch.matmul(attention_scores, self.query)  # shape: (batch_size, 1)
        attention_weights = F.softmax(attention_scores, dim=0)

        # Apply attention weights
        weighted_features = features * attention_weights.unsqueeze(-1)
        return weighted_features, attention_weights

# MultimodalClassifier_BA
class MultimodalClassifier_BA(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalClassifier_BA, self).__init__()
        # Image model
        self.image_model = EfficientNet.from_pretrained('efficientnet-b3')
        self.image_features_dim = self.image_model._fc.in_features
        self.image_model._fc = nn.Identity()  # We'll use the attention layer instead
        
        # Text model (BERT)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_features_dim = self.text_model.config.hidden_size

        self.image_attention = AttentionLayer(self.image_features_dim, 512)
        self.text_attention = AttentionLayer(self.text_features_dim, 512)

        # Final fusion layer
        self.fusion_linear = nn.Linear(self.image_features_dim + self.text_features_dim, num_classes)

    def forward(self, image_input, text_input_ids, text_attention_mask):
        # Process image features
        image_features = self.image_model(image_input)
        image_attended, image_weights = self.image_attention(image_features)

        # Process text features
        text_outputs = self.text_model(text_input_ids, attention_mask=text_attention_mask)
        pooled_text_output = text_outputs.last_hidden_state.mean(dim=1)
        text_attended, text_weights = self.text_attention(pooled_text_output)
        # print("Image attended shape:", image_attended.shape)
        # print("Text attended shape:", text_attended.shape)
        # Concatenate attended features as vectors
        #multimodal_features = torch.cat((image_attended.mean(dim=1), text_attended.mean(dim=1)), dim=1)
        multimodal_features = torch.cat((image_attended, text_attended), dim=1)
        # print("Multimodal features shape:", multimodal_features.shape)
        # Final classification or regression
        combined_logits = self.fusion_linear(multimodal_features)

        return combined_logits, image_weights, text_weights



class CrossModalSelfAttention(nn.Module):
    def __init__(self, text_feature_dim, image_feature_dim, attention_dim):
        super(CrossModalSelfAttention, self).__init__()
        self.text_query = nn.Linear(text_feature_dim, attention_dim)
        self.image_key = nn.Linear(image_feature_dim, attention_dim)
        self.image_value = nn.Linear(image_feature_dim, attention_dim)

    def forward(self, text_features, image_features):
        # Calculate queries, keys, values
        text_query = self.text_query(text_features)  # [batch_size, attention_dim]
        image_key = self.image_key(image_features)   # [batch_size, attention_dim]
        image_value = self.image_value(image_features) # [batch_size, attention_dim]
        
        # print("text_query shape:", text_query.shape)
        # print("image_key shape:", image_key.shape)
        # print("image_value shape:", image_value.shape)

        # Properly reshape for batch matrix multiplication
        text_query = text_query.unsqueeze(1)  # [batch_size, 1, attention_dim]
        image_key = image_key.unsqueeze(2)  # [batch_size, attention_dim, 1]

        # Calculate attention weights
        attention_weights = torch.bmm(text_query, image_key)  # [batch_size, 1, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # Apply softmax over the dimension of 'attention_dim'

        # Reshape image_value for applying attention
        image_value = image_value.unsqueeze(1)  # [batch_size, 1, attention_dim]

        # Apply attention weights to the values
        attended_image_features = torch.bmm(attention_weights, image_value)  # [batch_size, 1, attention_dim]
        attended_image_features = attended_image_features.squeeze(1)  # [batch_size, attention_dim]

        return attended_image_features + text_features, attention_weights.squeeze(-1).squeeze(-1)





#Cross Attention
class MultimodalClassifier_CA(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalClassifier_CA, self).__init__()
        self.image_model = EfficientNet.from_pretrained('efficientnet-b3')
        num_features = self.image_model._fc.in_features
        self.image_model._fc = nn.Identity()  # Remove final layer

        # Text Model (BERT)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_linear = nn.Linear(self.text_model.config.hidden_size, 512)

        # Attention Module
        self.cross_modal_attention = CrossModalSelfAttention(512, 1536, 512)

        # Fusion and Classification Layers
        # Ensure the input dimension here matches the concatenated features dimension
        self.fusion_linear = nn.Linear(2048, num_classes)

    def forward(self, image_input, text_input_ids, text_attention_mask):
        # Process image
        image_features = self.image_model(image_input)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten image features

        # Process text
        text_outputs = self.text_model(text_input_ids, attention_mask=text_attention_mask)
        pooled_text_output = text_outputs.pooler_output
        text_features = self.text_linear(pooled_text_output)

        # Apply Cross-modal Attention
        enhanced_text_features, attention_weights = self.cross_modal_attention(text_features, image_features)

        # Concatenate enhanced text and original image features
        multimodal_features = torch.cat((enhanced_text_features, image_features), dim=1)
        # print("multimodal_features shape:", multimodal_features.shape)
        combined_logits = self.fusion_linear(multimodal_features)

        return combined_logits, attention_weights
