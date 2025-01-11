import re

import torch

def load_checkpoint(model, checkpoint_path):
    """
    Function to load the pretrained model to continue training or evaluation
    :param model:
    :param checkpoint_path:
    :return:
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    match = re.search(r'_epoch(\d+).pth', checkpoint_path)
    if match:
        epoch_num = int(match.group(1))
        return model, checkpoint, epoch_num
    else:
        return model, checkpoint, None