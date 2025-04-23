import torch

class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
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
                    torch.nn.Linear(config.model.hidden_size, 1),
                )

    def forward(self, x, sigma=None):
        try:
            return self.backbone(x.float())
        except:
            raise ValueError(f"Incompatible shapes: {x.shape}")