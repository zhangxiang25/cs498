import sys
import os
import argparse

# --- Path Modification ---
# 1. Get the directory of this run.py script (e.g., C:\IsaacLab\cs498\project)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the root C:\IsaacLab directory (which is two levels up)
#    This assumes your folder structure is exactly C:\IsaacLab\cs498\project
isaaclab_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 3. Add the IsaacLab root to the Python path
#    This allows Python to find 'isaaclab.app'
if isaaclab_root not in sys.path:
    sys.path.append(isaaclab_root)

# 4. Add the current directory (where env.py is) to the Python path
#    This allows Python to find 'env.py' as a module
if current_dir not in sys.path:
    sys.path.append(current_dir)
# --- End Path Modification ---

# Now that the paths are set, we can import Isaac Lab
try:
    from isaaclab.app import launch_scene
except ImportError:
    print(f"Error: Could not import 'isaaclab.app'.")
    print(f"Please check that your IsaacLab root directory is correct: {isaaclab_root}")
    print("If it's not, please adjust the 'isaaclab_root' variable in this script.")
    sys.exit(1)


# The scene path is just "env:MP2SceneCfg" because 'env.py'
# is now directly in our Python path.
SCENE_CFG_PATH = "env:MP2SceneCfg"

def main():
    """Main function to launch the scene."""
    # Create a parser for command-line arguments
    parser = argparse.ArgumentParser(description="Run and visualize the custom project environment.")
    
    # Add the default Isaac Lab scene arguments
    launch_scene.add_cli_args(parser)
    
    # Parse the arguments
    args = parser.parse_args()

    # Launch the scene using the specified path and parsed arguments
    launch_scene(SCENE_CFG_PATH, cli_args=args)

if __name__ == "__main__":
    main()