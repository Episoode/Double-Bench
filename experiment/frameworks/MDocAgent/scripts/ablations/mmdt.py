import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mydatasets.base_dataset import BaseDataset
from agents.ablations import MDAt
import hydra

@hydra.main(config_path="../../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/"+agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/"+model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg
    
    cfg.mdoc_agent.sum_agent.agent = hydra.compose(config_name="agent/"+cfg.mdoc_agent.sum_agent.agent, overrides=[]).agent
    cfg.mdoc_agent.sum_agent.model = hydra.compose(config_name="model/"+cfg.mdoc_agent.sum_agent.model, overrides=[]).model
    
    dataset = BaseDataset(cfg.dataset)
    mdoc_agent = MDAt(cfg.mdoc_agent)
    mdoc_agent.predict_dataset(dataset)
    
if __name__ == "__main__":
    main()