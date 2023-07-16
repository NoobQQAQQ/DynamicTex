import torch


def pts2loss(pts1, pts2, weights, M=None):
    if M == None:
        loss = ((pts1 - pts2) ** 2).sum(-1)
        loss = (weights * loss).sum(-1)
        return loss.mean()
    else:
        loss = (((pts1 - pts2)*M[..., None]) ** 2).sum(-1)
        loss = (weights * loss).sum(-1).sum(0)
        # loss = (loss[..., None] * M).sum(0)
        return loss / (torch.sum(M) + 1e-8)


def img2mse(x, y, M=None):
    if M == None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8)


def img2mae(x, y, M=None):
    if M == None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x - y) * M) / (torch.sum(M) + 1e-8)


def L1(x, M=None):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + 1e-8)


def L2(x, M=None):
    if M == None:
        return torch.mean(x ** 2)
    else:
        return torch.sum((x ** 2) * M) / (torch.sum(M) + 1e-8)


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-19)) / x.shape[0]


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def NDC2world(pts, H, W, f):

    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1-1e-3) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def render_3d_point(H, W, f, pose, weights, pts):
    """Render 3D position along each ray and project it to the image plane.
    """

    c2w = torch.Tensor(pose)
    w2c = c2w[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])

    # Rendered 3D position in NDC coordinate
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)

    # NDC coordinate to world coordinate
    pts_map_world = NDC2world(pts_map_NDC, H, W, f)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[:, 3]
    # Rotate
    pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat([pts_map_cam[..., 0:1] / (- pts_map_cam[..., 2:]) * f + W * .5,
                          -pts_map_cam[..., 1:2] / (- pts_map_cam[..., 2:]) * f + H * .5],
                          dim=-1)

    return pts_plane


def induce_flow(H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d):

    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor = render_3d_point(H, W, focal,
                                      pose_neighbor,
                                      weights,
                                      pts_3d_neighbor)
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow
