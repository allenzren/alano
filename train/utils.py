from train import *
import os
import glob


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def save_model(model, step, logs_path, types, max_model=None):
    start = len(types) + 1
    os.makedirs(logs_path, exist_ok=True)
    if max_model is not None:
        model_list = glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > max_model - 1:
            min_step = min(
                [int(li.split('/')[-1][start:-4]) for li in model_list])
            os.remove(
                os.path.join(logs_path, '{}-{}.pth'.format(types, min_step)))
    logs_path = os.path.join(logs_path, '{}-{}.pth'.format(types, step))
    torch.save(model.state_dict(), logs_path)
    print('=> Save {} after [{}] updates'.format(logs_path, step))


# From https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385
# Had to move optimizer data to cpu so multiprocessing works
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
