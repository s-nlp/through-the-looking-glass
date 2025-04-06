import random
from enum import Enum

import numpy as np
import torch


class DatasetType(Enum):
    WEIRD = "weird"
    WHOOPS = "whoops"


LEARNING_RATE = 1e-4
BATCH_SIZE = 256
SEED = 14
DEFAULT_CHECKPOINT_PATH = "sileod/deberta-v3-large-tasksource-nli"
DEFAULT_EXTRACTED_FACTS_PATH = "../extracted_facts/5_facts_llava_mistral_16_diversity_1_0.jsonl"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
