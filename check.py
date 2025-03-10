import torch
print(torch.backends.mps.is_available())  # Should print True if MPS is supported
print(torch.backends.mps.is_built())  # Should also print True