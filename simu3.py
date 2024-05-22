import torch

def x_t(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    t = (10. * torch.sin(max(x1, x2, x3)) + max(x3, x4, x5)**3)/(1. + (x1 + x5)**2) + \
        torch.sin(0.5 * x3) * (1. + torch.exp(x4 - 0.5 * x3)) + x3**2 + 2. * torch.sin(x4) + 2.*x5 - 6.5
    return t

def x_t_link(t):
    return 1. / (1. + torch.exp(-1. * t))

def g_t(t, delta):
    return torch.cos((t-0.3) * 3.14159 * 2.) * delta * (t+0.2)**2 + 1

def t_x_y(t, x, delta):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    y = torch.cos((t-0.5) * 3.14159 * 2.) * (4.*max(x1, x6)**3)/(1. + 2.*x3**2) * torch.sin(x4-0.5) + g_t(t, delta)
    return y

def simu_data3(n_train, delta):
    train_matrix = torch.zeros(n_train, 8)

    for _ in range(n_train):
        x = torch.rand(6)
        train_matrix[_, 1:7] = x
        t = x_t(x)
        t += torch.randn(1)[0] * 0.5
        t = x_t_link(t)
        train_matrix[_, 0] = t

        y = t_x_y(t, x, delta)
        y += torch.randn(1)[0] * 0.5

        train_matrix[_, -1] = y

    t_grid = torch.zeros(2, n_train)
    t_grid[0, :] = train_matrix[:, 0].squeeze()
    t_grid[1, :] = g_t(t_grid[0, :], delta)

    return train_matrix, t_grid
