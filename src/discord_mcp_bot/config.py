import os
import yaml

# def env_constructor(loader, node):
#     """
#     Custom YAML constructor to load environment variables.
    
#     Args:
#         loader (yaml.Loader): The YAML loader.
#         node (yaml.Node): The YAML node.
    
#     Returns:
#         str: The value of the environment variable.
#     """
#     value = loader.construct_scalar(node)
#     env_var = os.environ.get(value)
#     if env_var is None:
#         raise ValueError(f"Environment variable '{value}' not found.")
#     return env_var

# yaml.add_constructor('!ENV', env_constructor)

def load_config(file_path):
    """
    Load the configuration file.
    
    Args:
        file_path (str): Path to the configuration file.
    
    Returns:
        dict: Configuration settings.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config