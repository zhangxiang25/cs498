import torch
import torch.utils.data as data
import torch.nn as nn


# TODO: dataset definition
class BCDataset (data.Dataset):

    def __init__ (self):
        pass
    

class Policy (nn.Module):

    def __init__(self):

        super(Policy, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # TODO: finish model definition

    
    def forward (self, image):

        # image will have shape (N, H, W, C), but needs to have shape (N, C, H, W)
        image = torch.swapaxes(image, 1, 3)
        image = torch.swapaxes(image, 2, 3)
        
        # encode image
        image /= 255.
        encoded_image = self.encoder(image)
        encoded_image = torch.flatten(encoded_image, 1)

        # TODO: predict action from encoded image
        action = None

        return action


def train_model (model, train_dataset, val_dataset):

    # TODO: model training loop
    pass


if __name__ == "__main__":

    # define and train model
    model = Policy()
    train_model(model)