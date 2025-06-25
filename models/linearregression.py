import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, config):
        super(LinearRegression, self).__init__()
        self.config = config
        if config.parameterization == "mine":
            raise NotImplementedError("Logistic Regression does not support 'mine' parameterization.")
        else:
            self.reg =  torch.nn.Linear(config.model.length, 3 * config.model.length)

    def forward(self, x, sigma=None):
        if self.config.parameterization == "mine":
            raise NotImplementedError("Logistic Regression does not support 'mine' parameterization.")
        else:
            output = self.reg(x.float())
            return output.reshape(-1, self.config.model.length, 3)