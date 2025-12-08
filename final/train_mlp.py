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

        # === MODIFIED ARCHITECTURE FOR DOOR TASK ===
        # Input: 8 dims 
        #   [Robot_X, Robot_Y, Robot_Z, Gripper_State, Door_X, Door_Y, Door_Z, Robot_Yaw]
        # Output: 11 dims
        #   [+X, -X, +Y, -Y, +Z, -Z, Open, Close, Stationary, Rot+, Rot-]
        
        self.net = nn.Sequential(
            nn.Linear(8, 256),  # Increased width slightly for more complex task
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 11)  # Output size = 11
        )

    
    def forward (self, x):
        return self.net(x)

def train_model (model, train_dataset, val_dataset):

    learning_rate = 1e-4
    batch_size = 32
    num_epochs = 500  # You can adjust this based on convergence

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    print("Calculating class weights for imbalance handling...")
    all_labels = np.argmax(train_dataset.actions, axis=1) # 把 One-hot 转成索引
    class_counts = np.bincount(all_labels, minlength=11)  # 统计每类有多少个
    
    print(f"Class counts: {class_counts}") # 打印出来看看分布情况
    # 1. 计算基础权重
    total_samples = len(all_labels)
    # 加上 1.0 防止除以 0，但后面我们会修正它
    class_weights = total_samples / (11 * (class_counts + 1.0))
    
    # 2. 【关键步骤】手动修复数量为 0 的类别
    # 找到数量为 0 的类别索引（比如你的 index 2）
    zero_count_indices = np.where(class_counts == 0)[0]
    
    # 将这些类别的权重强制设为 0.0 (让模型训练时忽略它，而不是报错或爆炸)
    class_weights[zero_count_indices] = 0.0 
    
    # 3. 转为 Tensor
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Safe Class Weights: {class_weights}")
    # 3. 将权重传给 Loss 函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_losses = []
    val_losses = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for obs, acts in train_loader:
            obs = obs.to(device).float()
            
            # Convert One-Hot Action Vectors to Class Indices
            # acts shape: [batch, 11] -> target_indices shape: [batch]
            target_indices = torch.argmax(acts, dim=1).to(device)

            optimizer.zero_grad()
            outputs = model(obs) 
            
            loss = criterion(outputs, target_indices)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * obs.size(0)

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
                target_indices = torch.argmax(acts, dim=1).to(device)

                outputs = model(obs)
                
                loss = criterion(outputs, target_indices)
                running_val_loss += loss.item() * obs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += target_indices.size(0)
                val_correct += (predicted == target_indices).sum().item()
        
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        val_acc = 100 * val_correct / val_total

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), 'mlp_model_door.pth') # 覆盖保存最好的
            # print(f"Saved best model with Acc: {val_acc:.2f}%")
    # Plotting
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_validation_loss_door.png') 

    # Save the model
    save_path = 'mlp_model_door.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # === PATH CONFIGURATION ===
    # Make sure this matches the path in collect_demo_door.py
    dataset_dir = r'C:\IsaacLab\cs498\final\image'

    all_observations = []
    all_actions = []

    print(f"Looking for data in: {dataset_dir}")
    file_list = glob.glob(os.path.join(dataset_dir, 'demo_*', 'states.npz'))

    if not file_list:
        print("No data found! Please run collect_demo_door.py first.")
        exit()
    else:
        print(f"Found {len(file_list)} episodes.")

    for file_path in file_list:
        try:
            data_map = np.load(file_path)
            observations = data_map['state_observations']
            actions = data_map['actions']
            
            # Simple check to ensure data dimensions match our new protocol
            if observations.shape[1] != 8 or actions.shape[1] != 11:
                print(f"Skipping {file_path}: Dimension mismatch. Expected Obs=8, Act=11. Found Obs={observations.shape[1]}, Act={actions.shape[1]}")
                continue

            all_observations.append(observations)
            all_actions.append(actions)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not all_observations:
        print("No valid data loaded.")
        exit()

    all_observations = np.concatenate(all_observations, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    print(f"Total samples: {all_observations.shape[0]}")

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