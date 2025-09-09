# -*- coding: utf-8 -*-

from torch import nn, cat

class ImageComparisonModel_mk1(nn.Module):
    def __init__(self):
        super(ImageComparisonModel_mk1, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Layer 1: CNN [in=6, out=6, kernel=3]
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1),
            nn.ReLU(),
            # Layer 2: CNN [in=6, out=3, kernel=1]
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1),
            nn.ReLU(),
            # Layer 3: CNN [in=3, out=1, kernel=3]
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        # Layer 5: Linear Layer [..., 1]
        self.linear_layer = nn.Linear(in_features=1*20*20, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img_clean, img_other):
        # Stack images together: (batch_size, 6, H, W)
        combined_images = cat((img_clean, img_other), dim=1)

        # Pass through CNN layers
        cnn_output = self.cnn_layers(combined_images)

        # Flatten the output
        flattened_output = self.flatten(cnn_output)

        # Pass through linear layer
        linear_output = self.linear_layer(flattened_output)

        # Apply Sigmoid for regression output between 0 and 1
        output = self.sigmoid(linear_output)

        return output
    


class ImageComparisonModel_mk2(nn.Module):
    '''
    Version 2:
    * More channles.
    * Pooling layers.
    '''
    def __init__(self):
        super(ImageComparisonModel_mk2, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Layer 1: CNN [in=6, out=16, kernel=3]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # reduce spatial size 20x20 → 10x10

            # Layer 2: CNN [in=16, out=32, kernel=3]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10 → 5x5

            # Layer 3: CNN [in=32, out=64, kernel=3]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        
        # Layer 5: Linear Layer [..., 1]
        self.linear_layer = nn.Linear(in_features=64*5*5, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img_clean, img_other):
        # Stack images together: (batch_size, 6, H, W)
        combined_images = cat((img_clean, img_other), dim=1)

        # Pass through CNN layers
        cnn_output = self.cnn_layers(combined_images)

        # Flatten the output
        flattened_output = self.flatten(cnn_output)

        # Pass through linear layer
        linear_output = self.linear_layer(flattened_output)

        # Apply Sigmoid for regression output between 0 and 1
        output = self.sigmoid(linear_output)

        return output


class ImageComparisonModel_mk3(nn.Module):
    '''
    Version 3:
    * Slightly less channles.
    * BatchNorm.
    * More layers between poolings.
    '''
    def __init__(self):
        super(ImageComparisonModel_mk3, self).__init__()
        self.cnn_layers = nn.Sequential(
            # --- First block (6 -> 8 channels) ---
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x20 -> 10x10

            # --- Second block (8 -> 16 channels) ---
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10 -> 5x5

            # --- Third block (16 -> 32 channels) ---
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        
        self.flatten = nn.Flatten()
        
        # Layer 5: Linear Layer [..., 1]
        self.linear_layer = nn.Linear(in_features=32*5*5, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img_clean, img_other):
        # Stack images together: (batch_size, 6, H, W)
        combined_images = cat((img_clean, img_other), dim=1)

        # Pass through CNN layers
        cnn_output = self.cnn_layers(combined_images)

        # Flatten the output
        flattened_output = self.flatten(cnn_output)

        # Pass through linear layer
        linear_output = self.linear_layer(flattened_output)

        # Apply Sigmoid for regression output between 0 and 1
        output = self.sigmoid(linear_output)

        # Return size [batch_size]
        return output.squeeze(1)


class ImageComparisonModel_mk4(nn.Module):
    '''
    Version 4:
    * More channels.
    * Note - keep SmoothL1Loss.
    '''
    def __init__(self):
        super(ImageComparisonModel_mk4, self).__init__()
        self.cnn_layers = nn.Sequential(
            # --- First block (6 -> 16 channels) ---
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x20 -> 10x10

            # --- Second block (16 -> 24 channels) ---
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10 -> 5x5

            # --- Third block (24 -> 48 channels) ---
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        
        # Layer 5: Linear Layer [..., 1]
        self.linear_layer = nn.Linear(in_features=48*5*5, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img_clean, img_other):
        # Stack images together: (batch_size, 6, H, W)
        combined_images = cat((img_clean, img_other), dim=1)

        # Pass through CNN layers
        cnn_output = self.cnn_layers(combined_images)

        # Flatten the output
        flattened_output = self.flatten(cnn_output)

        # Pass through linear layer
        linear_output = self.linear_layer(flattened_output)

        # Apply Sigmoid for regression output between 0 and 1
        output = self.sigmoid(linear_output)

        # Return size [batch_size]
        return output.squeeze(1)


class ImageComparisonModel_mk5(nn.Module):
    '''
    Version 5:
    * More layers.
    '''
    def __init__(self):
        super(ImageComparisonModel_mk5, self).__init__()
        self.cnn_layers = nn.Sequential(
            # --- First block (6 -> 8 channels) ---
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x20 -> 10x10

            # --- Second block (8 -> 16 channels) ---
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10 -> 5x5

            # --- Third block (16 -> 32 channels) ---
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 5x5 -> 2x2

            # --- Fourth block (32 -> 48 channels) ---
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        
        self.flatten = nn.Flatten()
        
        # Layer 5: Linear Layer [..., 1]
        self.linear_layer = nn.Linear(in_features=48*2*2, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, img_clean, img_other):
        # Stack images together: (batch_size, 6, H, W)
        combined_images = cat((img_clean, img_other), dim=1)

        # Pass through CNN layers
        cnn_output = self.cnn_layers(combined_images)

        # Flatten the output
        flattened_output = self.flatten(cnn_output)

        # Pass through linear layer
        linear_output = self.linear_layer(flattened_output)

        # Apply Sigmoid for regression output between 0 and 1
        output = self.sigmoid(linear_output)

        # Return size [batch_size]
        return output.squeeze(1)



class ImageComparisonModel_mk6(nn.Module):
    '''
    Version 6: Like version 4.
    * No Sigmoid.
    '''
    def __init__(self):
        super(ImageComparisonModel_mk6, self).__init__()
        self.cnn_layers = nn.Sequential(
            # --- First block (6 -> 16 channels) ---
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 20x20 -> 10x10

            # --- Second block (16 -> 24 channels) ---
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10x10 -> 5x5

            # --- Third block (24 -> 48 channels) ---
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        
        self.linear_layer = nn.Linear(in_features=48*5*5, out_features=1)

    def forward(self, img_clean, img_other):
        # Stack images together: (batch_size, 6, H, W)
        combined_images = cat((img_clean, img_other), dim=1)

        # Pass through CNN layers
        cnn_output = self.cnn_layers(combined_images)

        # Flatten the output
        flattened_output = self.flatten(cnn_output)

        # Pass through linear layer
        output = self.linear_layer(flattened_output)

        # Return size [batch_size]
        return output.squeeze(1)
    

# ============== MARK 7 ==============


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection: Adjusts input channels if they don't match the output
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Store the original input for the skip connection
        identity = x
        
        # Pass through the main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the shortcut connection
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

        
class ImageComparisonModel_mk7(nn.Module):
    '''
    Version 7:
    * More channels.
    * Residual blocks (ResNet).
    * Channel bottlenecking.
    * Kaiming weight init.
    '''
    def __init__(self):
        super(ImageComparisonModel_mk7, self).__init__()
        
        # Initial convolutional layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2) # 20x20 -> 10x10
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(16, 32, stride=2), # 10x10 -> 5x5
            ResidualBlock(32, 64),
            # ResidualBlock(64, 64),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # to handle various input sizes
        
        self.flatten = nn.Flatten()

        self.linear_layer = nn.Linear(in_features=64, out_features=1)
        
        # Kaiming weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)

    def forward(self, img_clean, img_other):
        # Stack images together: (batch_size, 6, H, W)
        combined_images = cat((img_clean, img_other), dim=1)

        # Pass through the initial convolutional layer
        x = self.initial_conv(combined_images)
        
        # Pass through residual blocks
        x = self.res_blocks(x)

        # Use global average pooling before the final linear layer
        x = self.avg_pool(x)

        # Flatten the output
        flattened_output = self.flatten(x)

        # Pass through the linear layer
        output = self.linear_layer(flattened_output)

        # Return size [batch_size]
        return output.squeeze(1)