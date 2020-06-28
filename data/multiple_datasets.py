import random
from torch.utils.data.dataset import Dataset

class MultipleDatasets(Dataset):
    def __init__(self, dbs):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])

    def __len__(self):
        return self.max_db_data_num * self.db_num

    def __getitem__(self, index):
        db_idx = index // self.max_db_data_num
        data_idx = index % self.max_db_data_num 
        if data_idx > len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])):
            data_idx = random.randint(0,len(self.dbs[db_idx])-1)
        else:
            data_idx = data_idx % len(self.dbs[db_idx])

        return self.dbs[db_idx][data_idx]
