import os, random, numpy as np

def seed_everything(seed: int | None = 42):
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)