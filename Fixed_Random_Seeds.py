# Fixed Random Seeds: CAI421 Fashion-MNIST Occlusion Project
# Mirrors the configuration in Cell 7 of Final_FashionMnist.ipynb.
# Cell 4 also uses random_state=42 in train_test_split.
# Running this as a separate shell process does NOT seed a running Colab kernel.
# The notebook already seeds its own kernel; this file documents/reuses those settings.

import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Random Seed:", SEED)
print("cuDNN deterministic:", torch.backends.cudnn.deterministic)
print("cuDNN benchmark:", torch.backends.cudnn.benchmark)

# Notes:
# - This matches the notebook's seed configuration but does not guarantee
#   identical results across all GPU models and library versions.
# - To reuse this file in Colab, run it in the SAME Python kernel,
#   before creating DataLoaders and the CNN model:
#       %run "Fixed_Random_Seeds.py"
#   (You do not need to run it when using the notebook's own Cell 7.)
