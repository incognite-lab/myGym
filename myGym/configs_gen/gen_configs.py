import glob
import os
import re
import random
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
GEN_CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs_gen")
SEQUENCES_PATH = os.path.join(GEN_CONFIGS_DIR, "action_list.yaml")

HOUSEHOLD_URDF_DIR = os.path.join(PROJECT_ROOT, "envs/objects/household/urdf")
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "AG_predicates.json")


ACTION_TO_CODE = {
    "approach": "A",
    "withdraw": "W",
    "grasp": "G",
    "drop": "D",
    "move": "M",
    "rotate": "R",
    "transform": "T",
    "follow": "F",
}

TASK_TYPES = [
    "A",
    "W",
    "AG",
    "AW",
    "AGD",
    "AGM",
    "AGR",
    "AGRD",
    "AGMD",
    "AGDW",
    "AGMDW",
    "AGRDW",
    "AGMDA",
]

def action_to_code(action: str) -> str:
    action = action.strip().lower()

    if action not in ACTION_TO_CODE:
        raise ValueError(f"Unknown action: {action}")

    return ACTION_TO_CODE[action]


def random_task_type_from_yaml(yaml_path: str, sequence_length: int) -> str:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if sequence_length not in data:
        raise ValueError(f"No sequences found for length {sequence_length}")

    possible_sequences = data[sequence_length]

    if not possible_sequences:
        raise ValueError(f"Sequence list for length {sequence_length} is empty")

    selected_sequence = random.choice(possible_sequences)
    action_list = selected_sequence["action_list"]

    task_type = "".join(action_to_code(action) for action in action_list)

    return task_type


def delete_json_files(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")


def replace_task_type(config_text: str, task_type: str) -> str:
    return re.sub(
        r'("task_type"\s*:\s*)"[A-Za-z0-9_]+"',
        rf'\1"{task_type}"',
        config_text,
        count=1,
    )


def generate_random_task_config(
    input_config_path: str,
    output_dir: str,
    yaml_path: str,
    sequence_length: int,
    clear_output_dir: bool = True,
) -> str:
    return

    os.makedirs(output_dir, exist_ok=True)

    if clear_output_dir:
        delete_json_files(output_dir)

    task_type = random_task_type_from_yaml(yaml_path, sequence_length)

    with open(input_config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    new_config_text = replace_task_type(config_text, task_type)

    i = 1
    while True:
        output_path = os.path.join(output_dir, f"gen_{task_type}{i}.json")

        if not os.path.exists(output_path):
            break

        i += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_config_text)

    print(f"Selected task type: {task_type}")
    print(f"Saved: {output_path}")

    return output_path


def get_household_object_names(urdf_dir: str = HOUSEHOLD_URDF_DIR) -> list:
    urdf_paths = sorted(glob.glob(os.path.join(urdf_dir, "*.urdf")))

    return [
        os.path.splitext(os.path.basename(path))[0]
        for path in urdf_paths
        if "target" not in os.path.basename(path)
    ]


def get_init_obj_name(config_text: str) -> str:
    match = re.search(
        r'"init"\s*:\s*\{\s*"obj_name"\s*:\s*"([A-Za-z0-9_]+)"', config_text
    )

    if not match:
        raise ValueError("Could not find task_objects init obj_name in config")

    return match.group(1)


def replace_object_name(config_text: str, old_name: str, new_name: str) -> str:
    return re.sub(
        rf"\b{re.escape(old_name)}\b", lambda _: new_name, config_text
    )


def generate_init_object_configs(input_config_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    delete_json_files(output_dir)

    with open(input_config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    init_obj_name = get_init_obj_name(config_text)

    for obj_name in get_household_object_names():
        new_config_text = replace_object_name(config_text, init_obj_name, obj_name)

        output_path = os.path.join(output_dir, f"gen_init_{obj_name}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_config_text)

        print(f"Saved: {output_path}")


def generate_task_configs(input_config_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    delete_json_files(output_dir)

    with open(input_config_path, "r", encoding="utf-8") as f:
        config_text = f.read()

    for task_type in TASK_TYPES:
        new_config_text = replace_task_type(config_text, task_type)

        i = 1
        while True:
            output_path = os.path.join(output_dir, f"gen_{task_type}{i}.json")

            if not os.path.exists(output_path):
                break

            i += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_config_text)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    reference_file = os.path.join(CONFIGS_DIR, "AGMDWa.json")
    output_folder = GEN_CONFIGS_DIR

    generate_random_task_config(
        input_config_path=reference_file,
        output_dir=output_folder,
        yaml_path=SEQUENCES_PATH,
        sequence_length=2,
    )

    generate_task_configs(reference_file, output_folder)

    generate_init_object_configs(
        input_config_path=DEFAULT_CONFIG,
        output_dir=os.path.join(GEN_CONFIGS_DIR, "grasp_objects"),
    )