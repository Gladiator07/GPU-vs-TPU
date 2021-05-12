import torch
dic = {
    "targets": torch.tensor([1,2,3,4]),
    "images": torch.tensor([[1,2,3], [3,4,5], [5,6,7]])
}


print(dic)
print(dic['targets'])
