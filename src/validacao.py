import torch
state_dict = torch.load(r"E:\Documents\lulc_env\tcc-fei\models\deeplabv3_best_fold1.pth", map_location="cpu")
print(list(state_dict.keys())[:20])  # Mostra as primeiras 20 chaves
