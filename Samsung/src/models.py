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
