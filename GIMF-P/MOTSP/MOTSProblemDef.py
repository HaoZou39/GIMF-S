import torch

def get_random_problems(batch_size, problem_size, num_objectives=2):
    """
    Generate random problem instances.
    
    Args:
        batch_size: Number of problem instances
        problem_size: Number of nodes
        num_objectives: Number of objectives (1 for single-objective, 2+ for multi-objective)
    
    Returns:
        problems: torch.Tensor
            - If num_objectives == 1: shape (batch_size, problem_size, 2)
              Single pair of coordinates [x, y] per node
            - If num_objectives >= 2: shape (batch_size, problem_size, 2*num_objectives)
              Multiple pairs [x1, y1, x2, y2, ...] per node
    """
    if num_objectives == 1:
        # Single objective: only generate one pair of coordinates [x, y]
        coord_dim = 2
    else:
        # Multi-objective: generate coordinates for each objective
        coord_dim = 2 * num_objectives
    
    problems = torch.rand(size=(batch_size, problem_size, coord_dim))
    return problems

def augment_xy_data_by_8_fold(xy_data, num_objectives=1):
    """
    Augment data by 8-fold using symmetry transformations.
    Applies to each objective space independently.
    
    Args:
        xy_data: (batch, nodes, 2*num_objectives)
        num_objectives: number of objectives
    
    Returns:
        Augmented data: (batch*8, nodes, 2*num_objectives)
    """
    # Extract coordinates for each objective
    coords = []
    for obj_idx in range(num_objectives):
        x = xy_data[:, :, [2*obj_idx]]      # x coordinate
        y = xy_data[:, :, [2*obj_idx + 1]]  # y coordinate
        coords.append((x, y))
    
    # Generate 8 augmentations for each objective
    aug_transforms = []
    for obj_idx in range(num_objectives):
        x, y = coords[obj_idx]
        dat = {}
        dat[0] = torch.cat((x, y), dim=2)
        dat[1] = torch.cat((1-x, y), dim=2)
        dat[2] = torch.cat((x, 1-y), dim=2)
        dat[3] = torch.cat((1-x, 1-y), dim=2)
        dat[4] = torch.cat((y, x), dim=2)
        dat[5] = torch.cat((1-y, x), dim=2)
        dat[6] = torch.cat((y, 1-x), dim=2)
        dat[7] = torch.cat((1-y, 1-x), dim=2)
        aug_transforms.append(dat)
    
    # Combine augmentations
    dat_aug = []
    for i in range(8):
        # Concatenate all objectives with the same transformation index
        obj_list = [aug_transforms[obj_idx][i] for obj_idx in range(num_objectives)]
        dat = torch.cat(obj_list, dim=2)
        dat_aug.append(dat)
    
    aug_problems = torch.cat(dat_aug, dim=0)
    return aug_problems


def augment_xy_data_by_64_fold_2obj(xy_data):
    """
    Legacy function for 2-objective 64-fold augmentation.
    Kept for backward compatibility.
    """
    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]

    dat1 = {}
    dat2 = {}

    dat_aug = []

    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1]= torch.cat((1-x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1-y1), dim=2)
    dat1[3] = torch.cat((1-x1, 1-y1), dim=2)
    dat1[4]= torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1-y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1-x1), dim=2)
    dat1[7] = torch.cat((1-y1, 1-x1), dim=2)

    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1]= torch.cat((1-x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1-y2), dim=2)
    dat2[3] = torch.cat((1-x2, 1-y2), dim=2)
    dat2[4]= torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1-y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1-x2), dim=2)
    dat2[7] = torch.cat((1-y2, 1-x2), dim=2)

    for i in range(8):
        for j in range(8):
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)

    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems


