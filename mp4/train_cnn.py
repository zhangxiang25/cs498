
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
import cv2

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def plot_learning_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve_CNN.png") 
    print("Plot saved to loss_curve.png")
    plt.show()


# TODO: dataset definition
class BCDataset (data.Dataset):

    def __init__ (self, dataset_dir, is_train=True):
        # Load all data from dataset_dir
        demo_folders =glob.glob(os.path.join(dataset_dir, "demo_*"))
        self.images = []

        for folder in demo_folders:
            data_file = os.path.join(folder, "states.npz")
            if not os.path.exists(data_file):
                continue    
            data = np.load(data_file)
            actions = data['actions']

            for i in range(len(actions)):
                img_path = os.path.join(folder, "images", f"{i}.png")
                if os.path.exists(img_path):
                    self.images.append((img_path, actions[i]))
        print(f"found {len(self.images)} samples in {len(demo_folders)} demos.")  
        self.is_train = is_train

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Shift image slightly
            transforms.ToTensor(), # Converts to [0, 1] automatically
        ])
        
        # Validation transform (just normalize/tensor)
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(), 
        ])
    
    def __len__ (self):
        return len(self.images)
    
    def __getitem__ (self, index):
        img_path, action = self.images[index]
        image = cv2.imread(img_path)
        image =cv2.resize(image, (256, 256))
        if self.is_train:
            image = self.transform(image)
        else:
            image = self.val_transform(image)
        # Convert action to tensor
        action = torch.tensor(action, dtype=torch.float32)
        return image, action

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
        self.flatten_size = 64*2*2

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 256),  # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5),                    # Drops 50% of neurons during training
            nn.Linear(256, 9)                   # Final action output
        )
    
    def forward (self, image):
        
        # encode image
        encoded_image = self.encoder(image)
        encoded_image = torch.flatten(encoded_image, 1)

        # TODO: predict action from encoded image
        action = self.classifier(encoded_image)

        return action


def train_model(model, train_dataset, val_dataset=None):
    # TODO: model training loop
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")
    
    model.to(device)
    
    # Create DataLoaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_dataset is not None:
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Loss and Optimizer
    # We use CrossEntropyLoss because the action is a classification problem (one key press at a time)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for i, (images, actions) in enumerate(train_loader):
            images = images.to(device)
            actions = actions.to(device) # Shape: (Batch, 9) (One-hot encoded)
            
            # Convert one-hot actions to class indices for CrossEntropyLoss
            # labels shape: (Batch,)
            target_classes = torch.argmax(actions, dim=1)
            
            # Forward pass
            outputs = model(images) # Shape: (Batch, 9)
            loss = criterion(outputs, target_classes)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_preds += target_classes.size(0)
            correct_preds += (predicted == target_classes).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds * 100
        train_loss.append(epoch_loss)

        val_epoch_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            
            with torch.no_grad():
                for images, actions in val_loader:
                    images = images.to(device)
                    actions = actions.to(device)
                    
                    target_classes = torch.argmax(actions, dim=1)
                    
                    outputs = model(images)
                    loss = criterion(outputs, target_classes)
                    
                    val_running_loss += loss.item()
                    
            val_epoch_loss = val_running_loss / len(val_loader)
            val_loss.append(val_epoch_loss)
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
        
    print("Training Complete.")
    
    # Save the model
    torch.save(model.state_dict(), "policy_model.pth")
    print("Model saved to policy_model.pth")
    return train_loss, val_loss

if __name__ == "__main__":
    data_dir = r'C:\IsaacLab\cs498\mp4\image'  # Path to dataset
    train_dataset = BCDataset(data_dir)
    # define and train model
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = data.random_split(train_dataset, [train_size, val_size])

    model = Policy()
    train_hist, val_hist = train_model(model, train_dataset, val_dataset)
    plot_learning_curve(train_hist, val_hist)
