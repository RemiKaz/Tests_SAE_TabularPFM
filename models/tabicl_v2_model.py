from models.tabicl_model import TabICL


class TabICLV2(TabICL):
    """TabICL v2, pinned to its checkpoint so a library upgrade does not change the model."""

    CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
