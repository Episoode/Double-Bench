from mydatasets.base_dataset import BaseDataset

class BaseRetrieval():
    def __init__(self, config):
        pass
    
    def prepare(self, dataset: BaseDataset):
        pass
    
    def find_top_k(self, dataset: BaseDataset):
        pass