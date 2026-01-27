#!/usr/bin/env python3
"""
TaskBuilder GUI for myGym

This script provides a graphical user interface for creating, editing, and managing
task configuration files. It loads the AG.json template and allows users to:
- Edit all parameters via dropdown menus
- Load existing configuration files
- Save configuration files
- Run train.py or test.py with the current configuration
"""

import os
import sys
import subprocess
import json
import commentjson
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path


class TaskBuilderGUI:
    """GUI for building and managing task configurations"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("myGym TaskBuilder")
        self.root.geometry("1000x800")
        
        # Get the myGym directory path
        self.mygym_dir = Path(__file__).parent
        self.config_dir = self.mygym_dir / "configs"
        self.template_path = self.config_dir / "AG.json"
        
        # Current configuration
        self.config = {}
        self.config_widgets = {}
        
        # Define available options for dropdown menus
        self.options = self._get_available_options()
        
        # Load template configuration
        self.load_template()
        
        # Create GUI
        self.create_widgets()
        
    def _get_available_options(self):
        """Define all available options for dropdown menus"""
        return {
            # Environment options
            "env_name": ["Gym-v0"],
            "workspace": ["table", "table_nico", "table_complex", "table_tiago"],
            "engine": ["pybullet"],
            "render": ["opengl", "opencv"],
            "camera": ["0", "1", "2", "3", "4"],
            "gui": ["0", "1"],
            "visualize": ["0", "1"],
            "visgym": ["0", "1"],
            
            # Robot options
            "robot": ["g1", "g1_full", "g1_rotslide", "gummi", "hsr", "jaco_gripper",
                     "kuka", "kuka_push", "kuka_gripper", "leachy", "nico", 
                     "panda", "panda_boxgripper", "panda_sgripper", "panda_gripper",
                     "pepper", "reachy", "tiago_single", "tiago_dual", "tiago_dual_fix",
                     "tiago_dual_rot", "tiago_dual_rotslide", "tiago_dual_rotslide2",
                     "ur3", "ur10", "yumi"],
            "robot_action": ["joints_gripper", "step_gripper", "absolute_gripper",
                           "joints", "step", "absolute"],
            "robot_init": ["default"],
            
            # Task type options
            "task_type": ["AG", "A", "AGM", "AGMD", "AGMDW", "AGRDW", "AGFDW", "AGR",
                         "reach", "push", "pnp", "dice_throw", "dropmag", "compositional"],
            "natural_language": ["0", "1"],
            
            # Object options (common objects)
            "object_names": ["null", "apple", "banana", "bottle", "bowl", "cube", "cube_small",
                           "sphere", "cylinder", "prism", "pyramid", "crackerbox", "fork",
                           "glass", "mug", "mustard", "plate", "spoon", "bar", "cube_holes",
                           "cube_target", "hammer", "peg_screw", "screwdriver", "sphere_holes",
                           "bird", "bus", "car", "dog", "duck", "helicopter", "horse", "plane",
                           "target", "bin", "box", "crate"],
            
            # Reward options
            "distance_type": ["euclidean", "manhattan"],
            
            # Training framework and algorithm
            "train_framework": ["pytorch"],
            "algo": ["ppo", "sac", "td3", "a2c", "multippo"],
            
            # Observation options
            "actual_state": ["obj_6D", "obj_xyz", "endeff_xyz", "endeff_6D", "vae", "yolact", "voxel", "dope"],
            "goal_state": ["obj_6D", "obj_xyz", "vae", "yolact", "voxel", "dope"],
            "additional_obs_options": ["joints_xyz", "joints_angles", "endeff_xyz", "endeff_6D", "touch", "distractor"],
        }
    
    def load_template(self):
        """Load the AG.json template configuration"""
        try:
            if not self.template_path.exists():
                raise FileNotFoundError(f"Template file not found: {self.template_path}")
            with open(self.template_path, 'r') as f:
                self.config = commentjson.load(f)
            print(f"Loaded template from: {self.template_path}")
        except FileNotFoundError as e:
            messagebox.showerror("Template Not Found", 
                               f"Cannot find AG.json template.\n\n{e}\n\nPlease ensure the template exists in the configs directory.")
            self.config = {}
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid Template", 
                               f"AG.json template has invalid JSON syntax.\n\n{e}")
            self.config = {}
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load template: {e}")
            self.config = {}
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Create main container with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add title
        title_label = ttk.Label(scrollable_frame, text="myGym Task Configuration Builder", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Create sections
        row = 1
        row = self.create_section(scrollable_frame, "Environment", row,
                                  ["env_name", "workspace", "engine", "render", "seed", 
                                   "camera", "gui", "visualize", "visgym"])
        
        row = self.create_section(scrollable_frame, "Robot", row,
                                  ["robot", "robot_action", "robot_init", "max_velocity", 
                                   "max_force", "action_repeat"])
        
        row = self.create_section(scrollable_frame, "Task", row,
                                  ["task_type", "natural_language"])
        
        row = self.create_section(scrollable_frame, "Task Objects", row,
                                  ["task_objects"])
        
        row = self.create_section(scrollable_frame, "Observation", row,
                                  ["observation"])
        
        row = self.create_section(scrollable_frame, "Reward", row,
                                  ["distance_type", "vae_path", "yolact_path", "yolact_config"])
        
        row = self.create_section(scrollable_frame, "Training", row,
                                  ["train_framework", "algo", "num_networks", "max_episode_steps",
                                   "algo_steps", "steps", "pretrained_model", "multiprocessing"])
        
        row = self.create_section(scrollable_frame, "Evaluation", row,
                                  ["eval_freq", "eval_episodes"])
        
        row = self.create_section(scrollable_frame, "Saving and Logging", row,
                                  ["record"])
        
        # Add buttons at the bottom (in the main window, not scrollable)
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Create two rows of buttons
        top_button_frame = ttk.Frame(button_frame)
        top_button_frame.pack(fill=tk.X, pady=(0, 5))
        
        bottom_button_frame = ttk.Frame(button_frame)
        bottom_button_frame.pack(fill=tk.X)
        
        # Top row: Load and Save
        load_btn = ttk.Button(top_button_frame, text="Load Config", command=self.load_config, width=20)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = ttk.Button(top_button_frame, text="Save Config", command=self.save_config, width=20)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Bottom row: Test and Train
        test_btn = ttk.Button(bottom_button_frame, text="Run Test", command=self.run_test, width=20)
        test_btn.pack(side=tk.LEFT, padx=5)
        
        train_btn = ttk.Button(bottom_button_frame, text="Run Train", command=self.run_train, width=20)
        train_btn.pack(side=tk.LEFT, padx=5)
        
    def create_section(self, parent, title, start_row, fields):
        """Create a section of configuration fields"""
        row = start_row
        
        # Section title
        section_label = ttk.Label(parent, text=title, font=('Arial', 12, 'bold'))
        section_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        row += 1
        
        # Add separator
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(0, 5))
        row += 1
        
        # Create fields
        for field in fields:
            if field in self.config:
                row = self.create_field(parent, field, self.config[field], row)
        
        return row
    
    def create_field(self, parent, field_name, field_value, row):
        """Create a single configuration field with label and widget"""
        # Label
        label = ttk.Label(parent, text=field_name + ":")
        label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Get help text from train.py comments if available
        help_text = self.get_help_text(field_name)
        if help_text:
            help_label = ttk.Label(parent, text="(?)", foreground="blue", cursor="hand2")
            help_label.grid(row=row, column=1, sticky=tk.W, padx=2)
            help_label.bind("<Button-1>", lambda e, text=help_text: messagebox.showinfo(field_name, text))
        
        # Widget based on field type
        if field_name in ["task_objects", "observation", "distractors", "color_dict", "used_objects"]:
            # Complex objects - use text widget
            widget = scrolledtext.ScrolledText(parent, height=4, width=60)
            widget.insert(1.0, json.dumps(field_value, indent=2))
            widget.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
        elif field_name in self.options:
            # Dropdown for predefined options
            widget = ttk.Combobox(parent, values=self.options[field_name], width=50)
            widget.set(str(field_value))
            widget.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
        elif isinstance(field_value, (int, float)):
            # Number entry
            widget = ttk.Entry(parent, width=50)
            widget.insert(0, str(field_value))
            widget.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
        elif isinstance(field_value, str) or field_value is None:
            # Text entry
            widget = ttk.Entry(parent, width=50)
            display_value = field_value if field_value is not None else "null"
            widget.insert(0, str(display_value))
            widget.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
        else:
            # Default: text widget
            widget = ttk.Entry(parent, width=50)
            widget.insert(0, str(field_value))
            widget.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
        
        self.config_widgets[field_name] = widget
        return row + 1
    
    def get_help_text(self, field_name):
        """Get help text for a field from train.py argument parser comments"""
        help_texts = {
            "env_name": "Environment name",
            "workspace": "Workspace name (table, table_nico, table_complex, table_tiago)",
            "engine": "Simulation engine name",
            "seed": "Seed number for reproducibility",
            "render": "Rendering type: opengl, opencv",
            "camera": "Number of cameras for rendering and recording",
            "visualize": "Visualize camera render and vision: 1 or 0",
            "visgym": "Visualize gym background: 1 or 0",
            "gui": "Use GUI: 1 or 0",
            "robot": "Robot to train: g1, panda, tiago, etc.",
            "robot_init": "Initial robot's end-effector position",
            "robot_action": "Robot's action control: step, absolute, joints (with optional _gripper suffix)",
            "max_velocity": "Maximum velocity of robotic arm",
            "max_force": "Maximum force of robotic arm",
            "action_repeat": "Substeps of simulation without action from env",
            "task_type": "Type of task to learn: AG (Approach+Grasp), AGM, reach, push, etc.",
            "task_objects": "Object (for reach) or pair of objects (for other tasks) to manipulate",
            "used_objects": "List of extra objects to randomly appear in the scene",
            "observation": "Observation space configuration with actual_state, goal_state, and additional_obs",
            "distance_type": "Type of distance metrics: euclidean, manhattan",
            "train_framework": "Training framework: pytorch",
            "algo": "Learning algorithm: ppo, sac, td3, a2c, multippo",
            "steps": "Number of training steps",
            "max_episode_steps": "Maximum number of steps per episode",
            "algo_steps": "Number of steps for algorithm training (PPO, A2C)",
            "eval_freq": "Evaluate the agent every eval_freq steps",
            "eval_episodes": "Number of episodes to evaluate performance",
            "record": "Recording: 0=no, 1=gif, 2=video",
            "multiprocessing": "Number of vectorized environments (for multiprocessing)",
            "pretrained_model": "Path to pretrained model to continue training",
        }
        return help_texts.get(field_name, "")
    
    def get_current_config(self):
        """Get current configuration from widgets"""
        config = {}
        for field_name, widget in self.config_widgets.items():
            try:
                if isinstance(widget, scrolledtext.ScrolledText):
                    # Parse JSON from text widget
                    value_str = widget.get(1.0, tk.END).strip()
                    config[field_name] = json.loads(value_str)
                elif isinstance(widget, ttk.Entry) or isinstance(widget, ttk.Combobox):
                    value_str = widget.get().strip()
                    # Try to parse as JSON, otherwise keep as string
                    if value_str.lower() == "null" or value_str == "":
                        config[field_name] = None
                    else:
                        try:
                            # Try to convert to appropriate type
                            if field_name in self.config and isinstance(self.config[field_name], int):
                                config[field_name] = int(value_str)
                            elif field_name in self.config and isinstance(self.config[field_name], float):
                                config[field_name] = float(value_str)
                            else:
                                config[field_name] = value_str
                        except ValueError:
                            config[field_name] = value_str
            except Exception as e:
                messagebox.showerror("Error", f"Error parsing field '{field_name}': {e}")
                return None
        return config
    
    def load_config(self):
        """Load a configuration file"""
        filename = filedialog.askopenfilename(
            initialdir=self.config_dir,
            title="Select configuration file",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    self.config = commentjson.load(f)
                
                # Update all widgets with loaded values
                for field_name, widget in self.config_widgets.items():
                    if field_name in self.config:
                        value = self.config[field_name]
                        if isinstance(widget, scrolledtext.ScrolledText):
                            widget.delete(1.0, tk.END)
                            widget.insert(1.0, json.dumps(value, indent=2))
                        elif isinstance(widget, ttk.Entry) or isinstance(widget, ttk.Combobox):
                            widget.delete(0, tk.END)
                            widget.insert(0, str(value) if value is not None else "null")
                
                messagebox.showinfo("Success", f"Configuration loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def save_config(self):
        """Save current configuration to a file"""
        filename = filedialog.asksaveasfilename(
            initialdir=self.config_dir,
            title="Save configuration file",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )
        
        if filename:
            config = self.get_current_config()
            if config is not None:
                try:
                    with open(filename, 'w') as f:
                        json.dump(config, f, indent=4)
                    messagebox.showinfo("Success", f"Configuration saved to {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def run_test(self):
        """Run test.py with current configuration"""
        config = self.get_current_config()
        if config is None:
            return
        
        # Save temporary config
        temp_config_path = self.config_dir / "temp_taskbuilder.json"
        try:
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Run test.py
            test_script = self.mygym_dir / "test.py"
            cmd = [sys.executable, str(test_script), "--config", str(temp_config_path)]
            
            response = messagebox.askyesno("Run Test", 
                                          f"This will run:\n{' '.join(cmd)}\n\nContinue?")
            if response:
                # Run in terminal
                subprocess.Popen(cmd, cwd=str(self.mygym_dir))
                messagebox.showinfo("Test Started", "Test script has been started in a new process")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run test: {e}")
    
    def run_train(self):
        """Run train.py with current configuration"""
        config = self.get_current_config()
        if config is None:
            return
        
        # Save temporary config
        temp_config_path = self.config_dir / "temp_taskbuilder.json"
        try:
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Run train.py
            train_script = self.mygym_dir / "train.py"
            cmd = [sys.executable, str(train_script), "--config", str(temp_config_path)]
            
            response = messagebox.askyesno("Run Train", 
                                          f"This will run:\n{' '.join(cmd)}\n\nContinue?")
            if response:
                # Run in terminal
                subprocess.Popen(cmd, cwd=str(self.mygym_dir))
                messagebox.showinfo("Training Started", "Training script has been started in a new process")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run train: {e}")


def main():
    """Main entry point for the TaskBuilder GUI"""
    root = tk.Tk()
    app = TaskBuilderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
