import os

def get_file_from_package(filename, package, *subdirs):
    try:
        from importlib.resources import files
        try:
            main_file = files("mygym_ros").joinpath(
                filename
            )
        except ModuleNotFoundError:
            return None
    except ImportError:
        from pkg_resources import resource_filename
        try:
            main_file = resource_filename("mygym_ros", filename)
        except ModuleNotFoundError:
            main_file = resource_filename("myGym.ros.mygym_ros", filename)

    main_file = str(main_file)
    if not os.path.isfile(main_file):
        raise FileNotFoundError(f"Main file not found at {main_file}")

