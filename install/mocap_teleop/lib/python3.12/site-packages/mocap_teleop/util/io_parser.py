import tomllib
import argparse as arg

from ament_index_python.packages import get_package_share_directory
import os

def parse_config():
    pkg_dir = get_package_share_directory('mocap_teleop')
    config_path = os.path.join(pkg_dir, 'config', 'config.toml')
    with open(config_path, 'rb') as f:   # tomllib requires binary mode
        return tomllib.load(f)

def parse_network_config() -> dict:
    config = parse_config()
    
    return config['networking']

def parse_rigid_body_config() -> dict:
    config = parse_config()
    
    return config['rigid_bodies']

def parse_mocap_config() -> dict:
    config = parse_config()
    
    return config['mocap']

def parse_learning_config() -> dict:
    config = parse_config()

    return config['learning']

def parse_controller_config() -> dict:
    config = parse_config()

    return config['controller']

def parse_logging_config() -> dict:
    config = parse_config()
    cfg = dict(config['logging'])

    raw_path = cfg['logs_dir']
    if not os.path.isabs(raw_path):
        # Resolve relative paths to the package source directory so logs land
        # next to config/, data/, and log/ regardless of which machine runs this.
        # get_package_share_directory → <ws>/install/<pkg>/share/<pkg>
        # Four levels up → <ws>; then into src/<pkg>.
        pkg_share = get_package_share_directory('mocap_teleop')
        ws_root   = os.path.normpath(os.path.join(pkg_share, '..', '..', '..', '..'))
        cfg['logs_dir'] = os.path.join(ws_root, 'src', 'mocap_teleop', raw_path)

    return cfg
