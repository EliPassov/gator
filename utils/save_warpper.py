import torch


def save_version_aware(state, filename, old_format=True):
    # check if version is 1.6.0 or above
    if int(torch.__version__.split('.')[1]) >= 6:
        torch.save(state, filename, _use_new_zipfile_serialization=(not old_format))
    else:
        if not old_format:
            raise ValueError('cannot save net using new format, torch version is below 1.6.0')
        torch.save(state, filename)
