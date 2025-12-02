# TODO: trajectory visualization
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def visualize_trajectories(dataset_dir, output_file):

    # Check if the directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found at '{dataset_dir}'")
        return

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Data lists for calculating plot bounds
    all_x_data = []
    all_y_data = []
    all_z_data = []

    # Iterate through all demo folders
    demo_count = 0
    for entry in os.listdir(dataset_dir):
        demo_path = os.path.join(dataset_dir, entry)
        states_file = os.path.join(demo_path, "states.npz")

        # Check if it's a valid demo folder with states.npz
        if os.path.isdir(demo_path) and os.path.exists(states_file):
            print(f"Loading trajectory from: {states_file}")
            
            try:
                # Load the saved data
                data = np.load(states_file)
                # The end-effector position is the first 3 columns of state_observations
                eef_positions = data['state_observations'][:, :3]
            except Exception as e:
                print(f"Could not load or parse data from {states_file}. Error: {e}")
                continue

            if eef_positions.shape[0] < 2:
                print(f"Skipping {entry}: Trajectory is too short.")
                continue

            # Extract coordinates
            x = eef_positions[:, 0]
            y = eef_positions[:, 1]
            z = eef_positions[:, 2]

            # Store data for plot scaling
            all_x_data.append(x)
            all_y_data.append(y)
            all_z_data.append(z)

            # Use a light color for the line and let markers stand out
            ax.plot(x, y, z, label=f'Demo {demo_count}', linewidth=1.5, alpha=0.7)

            # Mark the Start Point (first point)
            ax.scatter(x[0:1], y[0:1], z[0:1], 
                       color='green', marker='o', s=100, depthshade=True,
                       label=f'Start (Demo {demo_count})' if demo_count == 0 else None,
                       edgecolors='black', zorder=10) # zorder to ensure it's on top

            # Mark the End Point (last point)
            ax.scatter(x[-1:], y[-1:], z[-1:], 
                       color='red', marker='o', s=80, depthshade=True,
                       label=f'End (Demo {demo_count})' if demo_count == 0 else None,
                       edgecolors='black', zorder=10)

            demo_count += 1
            
        else:
            pass

    if demo_count == 0:
        print(f"No valid demonstrations found in '{dataset_dir}'.")
        return
        
    # Set plot properties
    ax.set_title(f'Robot End-Effector Trajectories ({demo_count} Demos)')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')

    # Add a legend for the start/end markers (only need one set)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Start (Green Circle)', markersize=10, markerfacecolor='green', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', label='End (Red Circle)', markersize=10, markerfacecolor='red', markeredgecolor='black')
    ], loc='best', title="Legend")

    # Set aspect ratio to roughly equal
    if all_x_data:
        all_x = np.concatenate(all_x_data)
        all_y = np.concatenate(all_y_data)
        all_z = np.concatenate(all_z_data)
        
        max_range = np.array([all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]).max()
        
        # Calculate centers
        center_x = (all_x.max() + all_x.min()) * 0.5
        center_y = (all_y.max() + all_y.min()) * 0.5
        center_z = (all_z.max() + all_z.min()) * 0.5
        
        # Create invisible points for cubic box to force equal aspect ratio
        Xb = 0.5 * max_range * np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + center_x
        Yb = 0.5 * max_range * np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + center_y
        Zb = 0.5 * max_range * np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + center_z
        
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w', alpha=0)


    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot successfully saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving plot to {output_file}: {e}")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize robot end-effector trajectories from collected demonstrations.")
    # Setting the default dataset path to the path provided by the user.
    parser.add_argument("--dataset_dir", 
                        type=str, 
                        default=r'C:\IsaacLab\cs498\mp4\image', 
                        help="Path to the root directory containing the collected demonstrations (e.g., 'C:/IsaacLab/cs498/mp4/image').")
    
    # New argument for the output file
    parser.add_argument("--output_file", 
                        type=str, 
                        default='robot_trajectories.png', 
                        help="Path and filename to save the visualization plot (e.g., 'my_plot.pdf').")
    
    args = parser.parse_args()
    
    # Pass the new output_file argument to the function
    visualize_trajectories(args.dataset_dir, args.output_file)