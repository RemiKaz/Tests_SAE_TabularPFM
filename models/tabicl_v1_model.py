from models.tabicl_model import TabICL


class TabICLV1(TabICL):
    """TabICL v1, the checkpoint of the TabICLv1 paper. The architecture comes from the
    checkpoint config, so the same layers (`tf_col`, `tf_row`, `tf_icl`) are available."""

    CHECKPOINT_VERSION = "tabicl-classifier-v1-20250208.ckpt"
