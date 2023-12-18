from itertools import repeat
import torch


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader

def save_model(i, gen, dis, opt_gen, opt_dis, cfg):
    name_save = cfg['trainer']['save_path'] + f'/epoch_{i}.pth'
    state = {
            "epoch": i,
            "state_dict_gen": gen.state_dict(),
            "state_dict_dis": dis.state_dict(),
            "opt_gen": opt_gen.state_dict(),
            "opt_dis": opt_dis.state_dict()
        }
    torch.save(state, name_save)
    print(f'model_saved epoch={i}')