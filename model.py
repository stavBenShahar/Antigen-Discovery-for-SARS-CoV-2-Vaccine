from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size: int = 180, hidden_layers=None, output_size=1, with_activation_func: bool = True):
        """
        Initializes the MLP model.

        Parameters:
        input_size (int): The size of the input layer.
        hidden_layers (list): A list of integers where each integer represents the size of a hidden layer.
        output_size (int): The size of the output layer.
        with_activation_func (bool): If True, ReLU activation functions will be added after each linear layer.
        """
        super().__init__()

        self.network = nn.Sequential()
        self.network.add_module("Input_Layer", nn.Linear(input_size, hidden_layers[0]))

        if with_activation_func:
            self.network.add_module("Relu_Layer", nn.ReLU())

        for i in range(1, len(hidden_layers)):
            self.network.add_module(f"Linear_Layer_{i}", nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            if with_activation_func:
                self.network.add_module(f"Relu_Layer_{i}", nn.ReLU())

        self.network.add_module("Output_layer", nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        return self.network(x)
