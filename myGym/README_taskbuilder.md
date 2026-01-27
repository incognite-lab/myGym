# TaskBuilder GUI

A graphical user interface for creating, editing, and managing task configuration files in myGym.

## Features

- **Load Template**: Automatically loads AG.json as a template configuration
- **Edit Parameters**: All parameters are available as dropdown menus or text fields
- **Parameter Options**: Dropdown options are populated from:
  - Available robots in `envs/robots/`
  - Available workspaces
  - train.py parameters and their comments
  - Other configuration files
- **Load/Save**: Load existing configs or save current config
- **Run Scripts**: Execute test.py or train.py with the current configuration

## Usage

### Running the GUI

```bash
cd myGym
python3 myGym/taskbuilder.py
```

Or make it executable and run directly:

```bash
chmod +x myGym/taskbuilder.py
./myGym/taskbuilder.py
```

### Requirements

The TaskBuilder requires tkinter (Python's standard GUI library):

**On Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**On macOS:**
tkinter is included with Python by default

**On Windows:**
tkinter is included with Python by default

### Interface Overview

The GUI is organized into sections:

1. **Environment**: env_name, workspace, engine, render, camera, gui, etc.
2. **Robot**: robot type, robot_action, max_velocity, max_force, etc.
3. **Task**: task_type, natural_language
4. **Task Objects**: Complex JSON configuration for objects
5. **Observation**: actual_state, goal_state, additional_obs configuration
6. **Reward**: distance_type, vae_path, yolact_path, etc.
7. **Training**: train_framework, algo, steps, max_episode_steps, etc.
8. **Evaluation**: eval_freq, eval_episodes
9. **Saving and Logging**: record option

### Buttons

- **Load Config**: Open an existing JSON configuration file
- **Save Config**: Save the current configuration to a JSON file
- **Run Test**: Execute test.py with the current configuration
- **Run Train**: Execute train.py with the current configuration

### Dropdown Options

The following dropdowns are populated with available options:

- **Robots**: g1, panda, tiago_dual, nico, kuka, jaco, yumi, etc.
- **Workspaces**: table, table_nico, table_complex, table_tiago
- **Task Types**: AG, A, AGM, AGMD, reach, push, pnp, etc.
- **Algorithms**: ppo, sac, td3, a2c, multippo
- **Robot Actions**: joints_gripper, step_gripper, absolute_gripper, etc.
- **Objects**: apple, cube, sphere, banana, bottle, fork, etc.

### Help Text

Click the blue (?) icon next to any parameter to see help text explaining what the parameter does.

## Configuration Files

- **Template**: `configs/AG.json` - Default template loaded on startup
- **Saved Configs**: Can be saved anywhere, recommended in `configs/` directory
- **Temporary Config**: When running test/train, a temporary config `temp_taskbuilder.json` is created

## Examples

### Creating a New Configuration

1. Launch TaskBuilder
2. Modify parameters as needed using dropdowns
3. Click "Save Config" to save to a new file

### Editing an Existing Configuration

1. Launch TaskBuilder
2. Click "Load Config" and select your configuration file
3. Modify parameters as needed
4. Click "Save Config" to save changes

### Testing a Configuration

1. Configure parameters in the GUI
2. Click "Run Test" to launch test.py with your configuration
3. The test will run in a separate process

### Training with a Configuration

1. Configure parameters in the GUI
2. Click "Run Train" to launch train.py with your configuration
3. Training will run in a separate process
4. Monitor progress in the terminal/console

## Troubleshooting

### "ModuleNotFoundError: No module named 'tkinter'"

Install python3-tk:
```bash
sudo apt-get install python3-tk
```

### GUI doesn't appear

If running on a headless server, you may need to use a virtual display:
```bash
sudo apt-get install xvfb
xvfb-run python3 myGym/taskbuilder.py
```

### Configuration not loading

- Ensure the JSON file is valid
- Check that all required fields are present
- Look for error messages in the console

## Notes

- Complex fields (task_objects, observation, etc.) use JSON text editors
- All changes are validated before saving
- Running test/train creates a temporary config file
- The GUI does not block - you can run multiple tests/training sessions
