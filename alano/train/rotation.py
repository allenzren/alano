import torch
import torch.nn.functional as F
import numpy as np


def rotate_tensor(orig_tensor, theta):
    """
	Rotate images clockwise
	"""
    affine_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                           [-np.sin(theta), np.cos(theta), 0]])
    affine_mat.shape = (2, 3, 1)
    affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float()
    flow_grid = torch.nn.functional.affine_grid(affine_mat,
                                                orig_tensor.size(),
                                                align_corners=False)
    return torch.nn.functional.grid_sample(orig_tensor,
                                           flow_grid,
                                           mode='nearest',
                                           align_corners=False)


def SO3_from_6D_torch(a1, a2):
    N = a1.shape[0]

    b1 = F.normalize(a1, p=2, dim=1)
    b2 = F.normalize(
        a2 -
        torch.matmul(b1.view(N, 1, -1), a2.view(N, -1, 1)).squeeze(2) * b1,
        p=2,
        dim=1)
    b3 = torch.cross(b1, b2)

    return torch.cat((b1.unsqueeze(2), b2.unsqueeze(2), b3.unsqueeze(2)),
                     dim=2)


def vec2rot_torch(x, y, device):
    N = x.shape[0]
    skewTransform = torch.tensor(
        [[0, 0, 0], [0, 0, -1], [0, 1, 0], [0, 0, 1], [0, 0, 0], [-1, 0, 0],
         [0, -1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=torch.float).to(device)

    v = torch.cross(x, y).unsqueeze(2)
    # s = torch.norm(v, dim=1)
    c = (x * y).sum(1).unsqueeze(1)  # batch dot product
    vs = torch.matmul(skewTransform, v).view(N, 3, 3)

    eye_tensor = torch.eye(3).expand(N, 3, 3).to(device)

    return eye_tensor + vs + (1 / (1 + c))[:, None] * torch.matmul(vs, vs)
