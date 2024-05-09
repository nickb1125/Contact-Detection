import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, image_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1)
        self.resnet= models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze all layers except the last fc layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(1000, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.resnet(x))
        x = torch.relu(self.fc(x))
        x = x.view(x.size(0), -1)
        return x

class ContactNet(nn.Module):
    def __init__(self, image_size, input_size, hidden_size, num_layers, dropout=0.1):
        super(ContactNet, self).__init__()
        self.encoder1 = Encoder(image_size)
        self.encoder2 = Encoder(image_size)
        self.encoder3 = Encoder(image_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)  # Combined input size
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        encoded_output1 = self.encoder1(x1)
        encoded_output2 = self.encoder2(x2)
        encoded_output3 = self.encoder3(x3)
        
        # Concatenate the encoded outputs along the channel dimension
        combined_input = torch.stack((encoded_output1, encoded_output2, encoded_output3), dim=1)
        lstm_output, _ = self.lstm(combined_input)  # Add an additional dimension for sequence_length
        last_layer_output = lstm_output[:, -1, :]
        return self.sigmoid(self.linear(last_layer_output))