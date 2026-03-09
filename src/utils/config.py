"""Load experiment configuration from YAML."""
import os
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "experiment.yaml")


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()
RANDOM_STATE = CFG["random_state"]
TEST_SIZE = CFG["test_size"]
VALIDATION_SIZE = CFG["validation_size"]
RAW_DIR = os.path.join(PROJECT_ROOT, CFG["data"]["raw_dir"])
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CFG["data"]["processed_dir"])
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PROFILING_DIR = os.path.join(ARTIFACTS_DIR, "profiling")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
