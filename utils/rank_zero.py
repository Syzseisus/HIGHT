from lightning.pytorch.utilities import rank_zero_only


@rank_zero_only
def rank_zero_print(*args, **kwargs):
    print(*args, **kwargs)
