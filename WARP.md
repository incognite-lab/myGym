# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create and activate conda environment (Python 3.10 required)
conda create -n mygym python=3.10
conda activate mygym

# Install in development mode
pip install -e .

# Install system dependency if needed
sudo apt install libopenmpi-dev
```

### Testing and Visualization
```bash
# Test environment with default settings (visual preview)
python myGym/test.py

# Test with specific configuration
python myGym/test.py --config myGym/configs/debug.json

# Visualize robot without training
python myGym/visualize_robot.py

# Test robot inverse kinematics
python myGym/visualize_robot_ik.py
```

### Training
```bash
# Train with default settings
python myGym/train.py

# Train with specific configuration
python myGym/train.py --config myGym/configs/train.json

# Train with GUI enabled
python myGym/train.py --config myGym/configs/debug.json --gui 1

# Multi-processing training
python myGym/train.py --multiprocessing true

# Train from pretrained model
python myGym/train.py --pretrained_model path/to/model
```

### Task Validation and Analysis
```bash
# Run task checker (oraculum) - validates task completion
python myGym/oraculum.py

# Run task checker for specific configuration
python myGym/taskchecker.py --config myGym/configs/debug.json

# Evaluate results and compute averages
python myGym/eval_results_average.py

# Generate dataset for training
python myGym/generate_dataset.py

# Visualize training results
python myGym/visualize_results.py
```

### Multi-Configuration Testing
```bash
# Run multiple test configurations
python myGym/multitest.py
```

### Documentation
```bash
# Build documentation
cd docs
make html
```

## Code Architecture

### High-Level Structure

**myGym** is a modular robotics simulation toolkit built around Gymnasium environments with PyBullet physics. The architecture follows a plugin-based design where components can be mixed and matched.

### Core Components

#### 1. Environment System (`myGym/envs/`)
- **`gym_env.py`**: Main Gymnasium environment (`Gym-v0`) that orchestrates all components
- **`base_env.py`**: Base environment with camera and rendering capabilities
- **`robot.py`**: Robot abstraction layer supporting 14+ robot types (UR3/5/10, Kuka, Franka, etc.)
- **`task.py`**: Task definition system supporting multi-step manipulation tasks
- **`rewards.py`**: Modular reward system with protorewards for atomic behaviors

#### 2. Task and Reward Framework
- **Protorewards**: Atomic reward components (approach, grip, move, place) that can be composed
- **Multi-network support**: Can train multiple networks simultaneously and switch between them
- **Task composition**: Tasks defined by capital letter combinations (e.g., "AGM" = Approach+Grip+Move)

#### 3. Configuration System (`myGym/configs/`)
- JSON-based configuration with extensive customization options
- Separate configs for training, debugging, and specific robot-task combinations
- Support for natural language task descriptions

#### 4. Training Infrastructure (`myGym/stable_baselines_mygym/`)
- Custom Stable-Baselines3 implementations
- **Multi-PPO**: Can train multiple PPO networks within single task
- **Custom callbacks**: Specialized monitoring and evaluation
- **Vectorized environments**: Support for parallel training

#### 5. Vision and Perception (`myGym/envs/vision_module.py`)
- Multiple observation types: RGB, depth, semantic masks, object poses
- VAE and YOLACT integration for learned representations
- Camera system with multiple viewpoints

### Key Design Patterns

#### Modular Task Definition
Tasks are composed from atomic actions using a letter-based system:
- `A`: Approach object
- `G`: Grip/grasp object  
- `M`: Move object
- `D`: Place/drop object
- Multi-letter tasks (e.g., "AGMD") automatically determine number of sub-networks needed

#### Reward Composition
The `Protorewards` class allows atomic reward functions to be combined:
- Each capital letter in task_type corresponds to a reward component
- Networks automatically switch based on task progress
- Ground truth or keyboard-based network switching

#### Robot Abstraction
Single interface supports 14+ robots with different:
- DOF counts (6-20 joints)
- End effector types (grippers, magnets, passive palms)
- Control modes (absolute, step, joint space)

### Configuration Examples

**Simple reach task**:
```json
{
  "robot": "kuka",
  "task_type": "A",  // Approach only
  "workspace": "table",
  "observation": {"actual_state": "obj_xyz", "goal_state": "obj_xyz"}
}
```

**Multi-step pick-and-place**:
```json
{
  "robot": "nico_grasp", 
  "task_type": "AGMD",  // Approach+Grip+Move+Drop (4 networks)
  "algo": "multippo",
  "num_networks": 4
}
```

### Key Files for Extension
- Add new robots: Modify `myGym/envs/robot.py`
- Add new tasks: Inherit from `Protorewards` in `myGym/envs/rewards.py`
- Add new workspaces: Modify workspace definitions
- Add new observation types: Extend `myGym/envs/vision_module.py`

### Testing Strategy
- Always test with `python myGym/test.py` before training
- Use keyboard control (arrow keys, WASD, X/C for gripper) during testing
- The `oraculum.py` validates task completion automatically
- Configuration validation happens at environment initialization

This architecture enables rapid prototyping of robotic manipulation tasks by composing existing components rather than implementing from scratch.
