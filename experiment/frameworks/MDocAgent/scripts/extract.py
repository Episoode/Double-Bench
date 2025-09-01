import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    dataset = BaseDataset(cfg.dataset)
    dataset.extract_content()

if __name__ == "__main__":
    main()