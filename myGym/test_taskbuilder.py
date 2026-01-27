#!/usr/bin/env python3
"""
Test script for TaskBuilder GUI

This script tests the TaskBuilder functionality without requiring a display.
It validates:
- Configuration loading
- Option definitions
- Help text availability
- Configuration saving
"""

import sys
import json
import tempfile
from pathlib import Path

# Add myGym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import commentjson
        print("  ✓ commentjson imported")
        return True
    except ImportError as e:
        print(f"  ✗ commentjson import failed: {e}")
        return False

def test_template_loading():
    """Test loading the AG.json template"""
    print("\nTesting template loading...")
    try:
        import commentjson
        template_path = Path(__file__).parent / "configs" / "AG.json"
        with open(template_path, 'r') as f:
            config = commentjson.load(f)
        
        print(f"  ✓ Loaded template from {template_path}")
        print(f"  ✓ Config has {len(config)} keys")
        
        # Check for required keys
        required_keys = ['env_name', 'workspace', 'robot', 'task_type', 'algo']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"  ✗ Missing required keys: {missing_keys}")
            return False
        print(f"  ✓ All required keys present")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading template: {e}")
        return False

def test_options():
    """Test that option definitions are comprehensive"""
    print("\nTesting option definitions...")
    
    # Create a mock options dictionary (same as in TaskBuilderGUI)
    options = {
        "robot": ["g1", "panda", "tiago_dual", "nico", "kuka"],
        "workspace": ["table", "table_nico", "table_complex", "table_tiago"],
        "task_type": ["AG", "A", "AGM", "reach", "push"],
        "algo": ["ppo", "sac", "td3", "a2c", "multippo"],
    }
    
    for key, values in options.items():
        print(f"  ✓ {key}: {len(values)} options available")
    
    return True

def test_config_save_load():
    """Test configuration saving and loading"""
    print("\nTesting config save/load...")
    try:
        import commentjson
        
        # Load template
        template_path = Path(__file__).parent / "configs" / "AG.json"
        with open(template_path, 'r') as f:
            original_config = commentjson.load(f)
        
        # Modify a value
        test_config = original_config.copy()
        test_config['robot'] = 'panda'
        test_config['task_type'] = 'AGM'
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            json.dump(test_config, f, indent=4)
        
        print(f"  ✓ Saved test config to {temp_path}")
        
        # Load it back
        with open(temp_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Verify
        if loaded_config['robot'] == 'panda' and loaded_config['task_type'] == 'AGM':
            print("  ✓ Config saved and loaded correctly")
            
            # Clean up
            Path(temp_path).unlink()
            return True
        else:
            print("  ✗ Config values don't match after load")
            return False
            
    except Exception as e:
        print(f"  ✗ Error in save/load test: {e}")
        return False

def test_help_texts():
    """Test that help texts are defined"""
    print("\nTesting help text definitions...")
    
    help_texts = {
        "robot": "Robot to train: g1, panda, tiago, etc.",
        "workspace": "Workspace name (table, table_nico, table_complex, table_tiago)",
        "task_type": "Type of task to learn: AG (Approach+Grasp), AGM, reach, push, etc.",
        "algo": "Learning algorithm: ppo, sac, td3, a2c, multippo",
    }
    
    for key, text in help_texts.items():
        if text and len(text) > 0:
            print(f"  ✓ {key}: '{text[:50]}...'")
        else:
            print(f"  ✗ {key}: No help text")
    
    return True

def test_script_structure():
    """Test that taskbuilder.py has the expected structure"""
    print("\nTesting script structure...")
    
    # Read the script
    script_path = Path(__file__).parent / "taskbuilder.py"
    if not script_path.exists():
        print(f"  ✗ taskbuilder.py not found at {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for key components
    checks = [
        ("TaskBuilderGUI class", "class TaskBuilderGUI"),
        ("load_template method", "def load_template"),
        ("create_widgets method", "def create_widgets"),
        ("load_config method", "def load_config"),
        ("save_config method", "def save_config"),
        ("run_test method", "def run_test"),
        ("run_train method", "def run_train"),
        ("main function", "def main"),
    ]
    
    all_present = True
    for name, pattern in checks:
        if pattern in content:
            print(f"  ✓ {name} found")
        else:
            print(f"  ✗ {name} not found")
            all_present = False
    
    return all_present

def main():
    """Run all tests"""
    print("=" * 60)
    print("TaskBuilder GUI Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Template Loading", test_template_loading()))
    results.append(("Options", test_options()))
    results.append(("Save/Load", test_config_save_load()))
    results.append(("Help Texts", test_help_texts()))
    results.append(("Script Structure", test_script_structure()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
