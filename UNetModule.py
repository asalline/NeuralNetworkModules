import torch
import torch.nn as nn

class DoubleConvAndReLU(nn.Module):
    '''
    Class for double convolutions (2D) and ReLUs that are used in the standard U-Net architecture.
    '''
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:tuple, 
                 padding:int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.double_conv_and_ReLU = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU()
        )

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        return self.double_conv_and_ReLU(input_tensor)
    
class UNet(nn.Module):
    '''
    The U-Net architecture module. Very generalized and gives a modularity for the depth.
    This version goes from one input channel to 64 in the first convolution.
    '''
    def __init__(self, 
                in_channels:int,
                out_channels:int, depth:int,
                conv_kernel_size:tuple, 
                max_pool_kernel_size:tuple, 
                up_conv_kernel_size:tuple, 
                padding:int, 
                skip_connection_list:list) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.max_pool_kernel_size = max_pool_kernel_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.padding = padding
        self.skip_connection_list = skip_connection_list

        self.max_pools = nn.ModuleList()
        self.encoder_block_convs = nn.ModuleList()
        self.up_convolutions = nn.ModuleList()
        self.decoder_block_convs = nn.ModuleList()
        
        for k in range(self.depth):
            if k == 0:
                self.encoder_block_convs.append(DoubleConvAndReLU(self.in_channels, 64, self.conv_kernel_size, padding=self.padding))
                self.decoder_block_convs.append(DoubleConvAndReLU(self.out_channels, int(self.out_channels / 2), self.conv_kernel_size, padding=self.padding))
                self.up_convolutions.append(nn.ConvTranspose2d(self.out_channels, int(self.out_channels / 2), self.up_conv_kernel_size, stride=2, padding=0))
                self.max_pools.append(nn.MaxPool2d(self.max_pool_kernel_size))
                self.out_channels = int(self.out_channels / 2)
                self.in_channels = 64
            elif k < self.depth-1:
                self.encoder_block_convs.append(DoubleConvAndReLU(self.in_channels, int(self.in_channels * 2), self.conv_kernel_size, padding=self.padding))
                self.decoder_block_convs.append(DoubleConvAndReLU(self.out_channels, int(self.out_channels / 2), self.conv_kernel_size, padding=self.padding))
                self.up_convolutions.append(nn.ConvTranspose2d(self.out_channels, int(self.out_channels / 2), self.up_conv_kernel_size, stride=2, padding=0))
                self.max_pools.append(nn.MaxPool2d(self.max_pool_kernel_size))
                self.in_channels = int(self.in_channels * 2)
                self.out_channels = int(self.out_channels / 2)
            else:
                self.encoder_block_convs.append(DoubleConvAndReLU(self.in_channels, int(self.in_channels * 2), self.conv_kernel_size, padding=self.padding))
                self.decoder_block_convs.append(nn.Conv2d(self.out_channels, 1, self.conv_kernel_size, padding=self.padding))

    def forward(self, 
                input_tensor:torch.Tensor) -> torch.Tensor:
        
        residual_connection_tensor = input_tensor.clone()

        for k in range(self.depth-1):
            input_tensor = self.encoder_block_convs[k](input_tensor)
            self.skip_connection_list.append(input_tensor)
            input_tensor = self.max_pools[k](input_tensor)

        input_tensor = self.encoder_block_convs[-1](input_tensor)

        for i in range(self.depth-1):
            input_tensor = self.up_convolutions[i](input_tensor)
            input_tensor = torch.cat([input_tensor, self.skip_connection_list[-1]], dim=1)
            self.skip_connection_list.pop()
            input_tensor = self.decoder_block_convs[i](input_tensor)

        input_tensor = self.decoder_block_convs[-1](input_tensor)

        return input_tensor + residual_connection_tensor
