import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, image_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the fully connected layer dynamically based on the input image size
        conv_output_size = self._get_conv_output_size(image_size)
        self.fc = nn.Linear(conv_output_size, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _get_conv_output_size(self, image_size):
        # Assuming the input image size is image_size x image_size
        # Calculate the output size after passing through convolutional layers
        with torch.no_grad():
            x = torch.zeros(1, 5, image_size, image_size)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            conv_output_size = x.view(x.size(0), -1).size(1)
        return conv_output_size

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