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
            conv_output_size = x.view(1, -1).size(1)
        return conv_output_size

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, inputs):
        # inputs: [batch_size, sequence_length, input_size]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(inputs)  # lstm_out: [batch_size, sequence_length, hidden_size]
        
        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply fully connected layer for classification
        output = self.fc(last_output)  # [batch_size, 1]
        
        # Apply sigmoid activation for binary classification
        output = torch.sigmoid(output)  # [batch_size, 1]
        
        return output

class ContactNet(nn.Module):
    def __init__(self, image_size, input_size, hidden_size, num_layers, dropout=0.1):
        super(ContactNet, self).__init__()
        self.encoder1 = Encoder(image_size)
        self.encoder2 = Encoder(image_size)
        self.encoder3 = Encoder(image_size)
        self.encoder4 = Encoder(image_size)
        self.encoder5 = Encoder(image_size)
        
        self.lstm = LSTMModel(input_size * 5, hidden_size, num_layers, dropout)  # Combined input size
        
    def forward(self, x1, x2, x3, x4, x5):
        encoded_output1 = self.encoder1(x1)
        encoded_output2 = self.encoder2(x2)
        encoded_output3 = self.encoder3(x3)
        encoded_output4 = self.encoder4(x4)
        encoded_output5 = self.encoder5(x5)
        
        # Concatenate the encoded outputs along the channel dimension
        combined_input = torch.cat((encoded_output1, encoded_output2, encoded_output3, encoded_output4, encoded_output5), dim=1)
        
        lstm_output = self.lstm(combined_input.unsqueeze(1))  # Add an additional dimension for sequence_length
        return lstm_output