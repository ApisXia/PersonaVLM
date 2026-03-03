import yaml

# read config file
with open("configs/base_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)
