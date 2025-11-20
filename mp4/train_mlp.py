import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

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
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    
    def forward (self, x):
        return self.net(x)

def train_model (model, train_dataset, val_dataset):

    # TODO: model training loop
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 100

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for obs, acts in train_loader:
            # Move data to device and ensure type is Float
            obs = obs.to(device).float()
            
            # Convert One-Hot Action Vectors to Class Indices (REQUIRED for CrossEntropyLoss)
            target_indices = torch.argmax(acts, dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(obs) # Fixed: variable name from 'outpus' to 'outputs'
            
            # Calculate Loss: Pass outputs (logits) and target_indices (Long tensor)
            loss = criterion(outputs, target_indices)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * obs.size(0)

            # Calculate Accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += target_indices.size(0)
            train_correct += (predicted == target_indices).sum().item()

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_acc = 100 * train_correct / train_total

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for obs, acts in val_loader:
                obs = obs.to(device).float()
                # Calculate target indices for the validation batch
                target_indices = torch.argmax(acts, dim=1).to(device)

                outputs = model(obs) # Fixed: variable name from 'outpus' to 'outputs'
                
                # Calculate Loss: Pass outputs (logits) and target_indices (Long tensor)
                loss = criterion(outputs, target_indices)
                running_val_loss += loss.item() * obs.size(0)

                _, predicted = torch.max(outputs.data, 1) # Fixed: variable name from 'outpus' to 'outputs'
                val_total += target_indices.size(0)
                val_correct += (predicted == target_indices).sum().item()
        
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png') 

    torch.save(model.state_dict(), 'mlp_model.pth')
            
    pass


if __name__ == "__main__":
    # load data
    dataset_dir = r'C:\IsaacLab\cs498\mp4\image'

    all_observations = []
    all_actions = []

    file_list = glob.glob(os.path.join(dataset_dir, 'demo_*', 'states.npz'))

    for file_path in file_list:
        data_map = np.load(file_path)
        observations = data_map['state_observations']
        actions = data_map['actions']

        all_observations.append(observations)
        all_actions.append(actions)

    all_observations = np.concatenate(all_observations, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    # split data into train and val
    num_samples = all_observations.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_idx = int(0.8 * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_obs, train_acts = all_observations[train_indices], all_actions[train_indices]
    val_obs, val_acts = all_observations[val_indices], all_actions[val_indices]

    train_dataset = BCDataset(train_obs, train_acts)
    val_dataset = BCDataset(val_obs, val_acts)



    # define and train model
    model = Policy()
    train_model(model, train_dataset, val_dataset)