import torch
import torch.utils.data as data
import torch.nn as nn


class BCDataset (data.Dataset):

    def __init__ (self, observations, actions):
        assert len(observations) == len(actions)
        self.observations = observations.copy()
        self.actions = actions.copy()
    
    def __len__ (self):
        return len(self.observations)

    def __getitem__ (self, index):
        return self.observations[index], self.actions[index]
    

class Policy (nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        # TODO: MLP model definition

    
    def forward (self, x):
        pass


def train_model (model, train_dataset, val_dataset):

    # TODO: model training loop
    pass


if __name__ == "__main__":

    # define and train model
    model = Policy()
    train_model(model)