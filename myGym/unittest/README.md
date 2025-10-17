# myGym Unit Tests

This directory contains unit tests for the myGym project.

## Available Tests

### test_robots.py
Tests robot URDF joint limit reachability for all robots in the robot dictionary.

**Usage:**
```bash
python3 myGym/unittest/test_robots.py [--gui]
```

Options:
- `--gui`: Visualize the tests in PyBullet GUI

### test_train_configs.py
Tests train.py with all configuration files in the `./configs` folder. This test runs the training script with a specified number of steps (default: 10000) for each config file and reports which configs trained successfully.

**Usage:**
```bash
# Test all configs with default settings (10000 steps, 1200s timeout)
python3 myGym/unittest/test_train_configs.py

# Test all configs with custom step count
python3 myGym/unittest/test_train_configs.py --steps 5000

# Test a specific config file
python3 myGym/unittest/test_train_configs.py --config train_A_nico.json

# Custom timeout per config (in seconds)
python3 myGym/unittest/test_train_configs.py --timeout 600
```

Options:
- `--steps STEPS`: Number of training steps to run (default: 10000)
- `--timeout TIMEOUT`: Timeout in seconds for each config test (default: 1200)
- `--config CONFIG`: Test only a specific config file (provide filename or path)

**Output:**
The test will:
1. Print each config name with a ✔ OK mark if training succeeds without errors
2. Print a summary table at the end showing:
   - Total configs tested
   - Number of successful configs
   - Number of failed configs
3. Display a detailed list of successfully trained configs
4. Display failed configs with error messages (if any)

**Requirements:**
All dependencies from `pyproject.toml` must be installed:
```bash
pip install -e .
```

### test_oraculum_configs.py
Tests the oraculum method with all configuration files in the `./configs` folder. This test runs test.py with oraculum control (`-ct oraculum`) for each config file and checks if the tasks can be successfully completed. The test runs a specified number of trials (default: 5) for each config and marks it as successful if at least a minimum number of trials succeed (default: 5).

**Usage:**
```bash
# Test all configs with oraculum method (5 trials per config, requires 5 successes)
python3 myGym/unittest/test_oraculum_configs.py

# Test with custom number of trials
python3 myGym/unittest/test_oraculum_configs.py --trials 10

# Test a specific config file
python3 myGym/unittest/test_oraculum_configs.py --config train_A.json

# Custom timeout per config (in seconds)
python3 myGym/unittest/test_oraculum_configs.py --timeout 300

# Custom minimum successful trials required
python3 myGym/unittest/test_oraculum_configs.py --min-success 3
```

Options:
- `--trials TRIALS`: Number of trials (eval_episodes) to run per config (default: 5)
- `--timeout TIMEOUT`: Timeout in seconds for each config test (default: 300)
- `--config CONFIG`: Test only a specific config file (provide filename or path)
- `--min-success MIN_SUCCESS`: Minimum number of successful trials required (default: 5)

**Output:**
The test will:
1. Print each config name with a ✔ OK mark if it passes the minimum success criteria
2. Print a summary table at the end showing:
   - Total configs tested
   - Number of successful configs
   - Number of failed configs
3. Display a detailed list of successfully tested configs with success counts
4. Display failed configs with success counts and error messages (if any)

**Requirements:**
All dependencies from `pyproject.toml` must be installed:
```bash
pip install -e .
```

## Running All Tests

To run all tests in this directory:
```bash
cd myGym/unittest
python3 test_robots.py
python3 test_train_configs.py
python3 test_oraculum_configs.py
```

## Notes

- Tests require all project dependencies to be installed
- Training tests can take significant time depending on the number of steps and configs
- Oraculum tests can take significant time depending on the number of trials and configs
- Failed tests will show error messages to help with debugging
- Tests return appropriate exit codes (0 for success, 1 for failures)
