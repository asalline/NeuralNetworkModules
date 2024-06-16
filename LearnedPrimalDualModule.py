import torch
import torch.nn as nn

class PrimalResNet(nn.Module):
    '''
    ResNet architecture for the primal step in Learned Primal-Dual algorithm.
    '''
    def __init__(
        self, 
        num_of_layers:int,
        in_channels_primal:int,
        mid_channels:int,
        out_channels:int,
        conv_kernel_size:tuple, 
        padding:int,
        num_parameters:int,
        alpha:float) -> None:
        super().__init__()
        
        self.num_of_layers = num_of_layers
        self.in_channels_primal = in_channels_primal
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.padding = padding
        self.num_parameters = num_parameters
        self.alpha = alpha
        
        self.primal_resnet = nn.ModuleList()
        
        for k in range(self.num_of_layers):
            if k == 0:
                self.primal_resnet.append(nn.Conv2d(in_channels=self.in_channels_primal,
                                                    out_channels=self.mid_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))
                self.primal_resnet.append(nn.PReLU(num_parameters=self.num_parameters,
                                                   init=self.alpha))
            elif k < self.num_of_layers-1:
                self.primal_resnet.append(nn.Conv2d(in_channels=self.mid_channels,
                                                    out_channels=self.mid_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))
                self.primal_resnet.append(nn.PReLU(num_parameters=self.num_parameters,
                                                   init=self.alpha))
            else:
                self.primal_resnet.append(nn.Conv2d(in_channels=self.mid_channels,
                                                    out_channels=self.mid_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))
                self.primal_resnet.append(nn.PReLU(num_parameters=self.num_parameters,
                                                   init=self.alpha))
                self.primal_resnet.append(nn.Conv2d(in_channels=self.mid_channels,
                                                    out_channels=self.out_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))
        
        self.primal_resnet = nn.Sequential(*self.primal_resnet)
                        
    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        output_tensor = self.primal_resnet(input_tensor)
        return output_tensor
    
class DualResNet(nn.Module):
    '''
    ResNet architecture for the dual step in Learned Primal-Dual algorithm.
    '''
    def __init__(
        self, 
        num_of_layers:int,
        in_channels_dual:int, 
        mid_channels:int,
        out_channels:int,
        conv_kernel_size:tuple, 
        padding:int,
        num_parameters:int,
        alpha:float) -> None:
        super().__init__()
        
        self.num_of_layers = num_of_layers
        self.in_channels_dual = in_channels_dual
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.padding = padding
        self.num_parameters = num_parameters
        self.alpha = alpha
        
        self.dual_resnet = nn.ModuleList()
                
        for k in range(self.num_of_layers):
            if k == 0:
                self.dual_resnet.append(nn.Conv2d(in_channels=self.in_channels_dual,
                                                    out_channels=self.mid_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))
                self.dual_resnet.append(nn.PReLU(num_parameters=self.num_parameters,
                                                   init=self.alpha))
            elif k < self.num_of_layers-1:
                self.dual_resnet.append(nn.Conv2d(in_channels=self.mid_channels,
                                                    out_channels=self.mid_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))
                self.dual_resnet.append(nn.PReLU(num_parameters=self.num_parameters,
                                                   init=self.alpha))
            else:
                # print('here')
                self.dual_resnet.append(nn.Conv2d(in_channels=self.mid_channels,
                                                    out_channels=self.mid_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))
                self.dual_resnet.append(nn.PReLU(num_parameters=self.num_parameters,
                                                   init=self.alpha))
                self.dual_resnet.append(nn.Conv2d(in_channels=self.mid_channels,
                                                    out_channels=self.out_channels,
                                                    kernel_size=self.conv_kernel_size,
                                                    padding=self.padding))

        self.dual_resnet = nn.Sequential(*self.dual_resnet)

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        output_tensor = self.dual_resnet(input_tensor)
        return output_tensor
    
class LearnedPrimalDualStep(nn.Module):
    '''
    One unrolled iteration (step) of the Learned Primal-Dual algorithm.
    '''
    def __init__(
        self, 
        num_of_layers:int,
        in_channels_primal:int, 
        in_channels_dual:int, 
        mid_channels:int,
        out_channels:int,
        conv_kernel_size:tuple, 
        padding:int,
        num_parameters:int,
        alpha:float,
        operator,
        adjoint_operator,
        operator_norm:float) -> None:
        super().__init__()
        
        self.num_of_layers = num_of_layers
        self.in_channels_primal = in_channels_primal
        self.in_channels_dual = in_channels_dual
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.padding = padding
        self.num_parameters = num_parameters
        self.alpha = alpha
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        
        self.primal_resnet = PrimalResNet(num_of_layers=self.num_of_layers,
                                          in_channels_primal=self.in_channels_primal,
                                          mid_channels=self.mid_channels,
                                          out_channels=self.out_channels,
                                          conv_kernel_size=self.conv_kernel_size,
                                          padding=self.padding,
                                          num_parameters=self.num_parameters,
                                          alpha=self.alpha)
        
        self.dual_resnet = DualResNet(num_of_layers=self.num_of_layers,
                                          in_channels_dual=self.in_channels_dual,
                                          mid_channels=self.mid_channels,
                                          out_channels=self.out_channels,
                                          conv_kernel_size=self.conv_kernel_size,
                                          padding=self.padding,
                                          num_parameters=self.num_parameters,
                                          alpha=self.alpha)
        
    def forward(self, input_tensor:torch.Tensor, input_tensor_2:torch.Tensor, input_tensor_3:torch.Tensor) -> tuple:
        sinogram = self.operator(input_tensor) / self.operator_norm
        update = torch.cat([input_tensor_3, sinogram, input_tensor_2], dim=1)
        input_tensor_3 = input_tensor_3 + self.dual_resnet(update)
        adjoint_evaluation = self.adjoint_operator(input_tensor_3) / self.operator_norm
        update = torch.cat([input_tensor, adjoint_evaluation], dim=1)
        input_tensor = input_tensor + self.primal_resnet(update)
        
        return input_tensor, input_tensor_3
    
class LearnedPrimalDual(nn.Module):
    '''
    The whole algorithm is created here. This class calls all the other classes to build the unrolled network.
    '''
    def __init__(
        self, 
        num_of_layers:int,
        in_channels_primal:int, 
        in_channels_dual:int, 
        mid_channels:int,
        out_channels:int,
        conv_kernel_size:tuple, 
        padding:int,
        num_parameters:int,
        alpha:float,
        operator,
        adjoint_operator,
        operator_norm:float,
        num_unrolled_iterations:int) -> None:
        super().__init__()
        
        self.num_of_layers = num_of_layers
        self.in_channels_primal = in_channels_primal
        self.in_channels_dual = in_channels_dual
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.padding = padding
        self.num_parameters = num_parameters
        self.alpha = alpha
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.num_unrolled_iterations = num_unrolled_iterations
    
        for k in range(self.num_unrolled_iterations):
            LPD_step = LearnedPrimalDualStep(num_of_layers=self.num_of_layers,
                                             in_channels_primal=self.in_channels_primal,
                                             in_channels_dual=self.in_channels_dual,
                                             mid_channels=self.mid_channels,
                                             out_channels=self.out_channels,
                                             conv_kernel_size=self.conv_kernel_size,
                                             padding=self.padding,
                                             num_parameters=self.num_parameters,
                                             alpha=self.alpha,
                                             operator=self.operator,
                                             adjoint_operator=self.adjoint_operator,
                                             operator_norm=self.operator_norm)
            setattr(self, f'step_{k}', LPD_step)
            
    def forward(self, input_tensor:torch.Tensor, input_tensor_2:torch.Tensor) -> torch.Tensor:
        
        input_tensor_3 = torch.zeros(input_tensor_2.shape)
        
        for k in range(self.num_unrolled_iterations):
            LPD_step = getattr(self, f'step_{k}')
            input_tensor, input_tensor_3 = LPD_step(input_tensor, input_tensor_2, input_tensor_3)
            
        return input_tensor
