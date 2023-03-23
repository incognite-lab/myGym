import argparse
import os


LANGUAGE_FILE_PATH = os.path.join('envs', 'examples', 'language.txt')
PREDEFINED_ARGS = [
    '--config configs/train_reach.json',
    '--config configs/train_pnp_4n_multitask4.json',
    '--config configs/train_swipe.json',
]


def get_parser():
    parser = argparse.ArgumentParser(description='Show examples of language descriptions for specified environments. '
                                                 'It\'s assumed that the file is launched from the same directory '
                                                 'as the train.py')
    parser.add_argument('-t', '--type', type=str, default='description',
                        help='Expected values are \"description\" for generating a task description or '
                             '\"new_tasks\" for generating new tasks from a given environment')
    parser.add_argument('-cmd', '--command', type=str, default='',
                        help='If passed and the --type is \"description\", then a description is generated based on a '
                             'created environment, otherwise the predefined commands are used. If the --type is '
                             '"new_tasks", then the argument can\' be omitted')
    parser.add_argument('-out', '--output', type=int, default=0,
                        help='Whether to print output of the train.py, either 0 or 1')
    return parser


def get_description(launch_command, args):
    os.system(launch_command + ' --gui 0 -nl description' + ('' if args.output == 1 else ' >/dev/null 2>&1'))
    with open(LANGUAGE_FILE_PATH, "r") as file:
        lines = file.readlines()

        if len(lines) != 1:
            msg = f'Invalid file structure: {LANGUAGE_FILE_PATH}. Given {len(lines)} lines, but the script was expecting only one'
            raise Exception(msg)

        return lines[0]


def get_new_tasks(launch_command, args):
    os.system(launch_command + ' --gui 0 -nl new_tasks' + ('' if args.output == 1 else ' >/dev/null 2>&1'))
    with open(LANGUAGE_FILE_PATH, "r") as file:
        lines = file.readlines()

        if len(lines) == 0:
            msg = f'File {LANGUAGE_FILE_PATH} doesn\'t contain any information'
            raise Exception(msg)

        return lines


def main():
    args = get_parser().parse_args()

    if args.type == 'description':
        if args.command != '':
            print(get_description(args.command, args))
        else:
            print('"Launch command (environment task)" -> "Language description" mappings:')
            for cmd in map(lambda a: 'python train.py ' + a, PREDEFINED_ARGS):
                print(cmd + ' -> ' + get_description(cmd, args))
    elif args.type == 'new_tasks':
        if args.command != '':
            print('\n'.join(get_new_tasks(args.command, args)))
        else:
            msg = f'Command hasn\'t been specified!'
            raise Exception(msg)
    else:
        msg = f'Unknown type: {args.type}'
        raise Exception(msg)


if __name__ == "__main__":
    main()
