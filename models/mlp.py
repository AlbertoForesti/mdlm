import torch

class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        if config.parameterization == "mine":
            self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(config.model.length, config.model.hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(config.model.hidden_size, config.model.hidden_size),
                    torch.nn.LeakyReLU(),
                )
        else:
            self.backbone = torch.nn.Sequential(
                    torch.nn.Linear(config.model.length, config.model.hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(config.model.hidden_size, config.model.hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(config.model.hidden_size, 3*config.model.length),
                )

    def forward(self, x, sigma=None):
        if self.config.parameterization == "mine":
            return self.backbone(x.float())
        else:
            output = self.backbone(x.float())
            return output.reshape(-1, self.config.model.length, 3)