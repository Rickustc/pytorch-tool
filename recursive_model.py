
import torch

def recursive_module_helper(model):
    for name, child in model.named_children():
        print(name)
        if isinstance(child, torch.nn.Module):
            recursive_module_helper(child)
            
if __name__ == "__main__":
    recursive_module_helper(model)