
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
import cv2


# TODO: dataset definition
class BCDataset (data.Dataset):

    def __init__ (self, dataset_dir):
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
    
    def __len__ (self):
        return len(self.images)
    
    def __getitem__ (self, index):
        img_path, action = self.images[index]
        image = cv2.imread(img_path)
        image =cv2.resize(image, (256, 256))
        image = torch.tensor(image, dtype=torch.float32)
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

        self.fc =nn.Linear(self.flatten_size, 9)
    
    def forward (self, image):

        # image will have shape (N, H, W, C), but needs to have shape (N, C, H, W)
        image = torch.swapaxes(image, 1, 3)
        image = torch.swapaxes(image, 2, 3)
        
        # encode image
        image /= 255.
        encoded_image = self.encoder(image)
        encoded_image = torch.flatten(encoded_image, 1)

        # TODO: predict action from encoded image
        action = self.fc(encoded_image)

        return action


def train_model(model, train_dataset, val_dataset=None):
    # TODO: model training loop
    
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")
    
    model.to(device)
    
    # Create DataLoaders
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Loss and Optimizer
    # We use CrossEntropyLoss because the action is a classification problem (one key press at a time)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    print("Training Complete.")
    
    # Save the model
    torch.save(model.state_dict(), "policy_model.pth")
    print("Model saved to policy_model.pth")

if __name__ == "__main__":
    data_dir = r'C:\IsaacLab\cs498\mp4\image'  # Path to dataset
    train_dataset = BCDataset(data_dir)
    # define and train model
    model = Policy()
    train_model(model, train_dataset)