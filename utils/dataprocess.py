import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils.dataset import train_transform
import random

class DatasetSplitter:
    def __init__(self,train_dataset, val_dataset, test_dataset, batch_size=8, seed=42):
        self.train_dataset= train_dataset
        self.val_dataset= val_dataset
        self.test_dataset= test_dataset
        self.batch_size=batch_size
        self.seed= seed
        self._set_seed()

        #Create a torch.Generator for reproducible shuffling
        g= torch.Generator()
        g.manual_seed(self.seed)

        #Worker initialization function for multi-worker dataloader
        def worker_init_fn(worker_id):
            np.random.seed(self.seed+worker_id)
            random.seed(self.seed+worker_id)

        self.train_loader= DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            num_workers=4,
            worker_init_fn=worker_init_fn
        )
        
        self.val_loader= DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle= False,
            num_workers=4,
            worker_init_fn=worker_init_fn
        )

        self.test_loader= DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=worker_init_fn
        )

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def print_summary(self):
        print(f"Train : {len(self.train_dataset)}, Val : {len(self.val_dataset)}, Test: {len(self.test_dataset)}")


