from .base import BaseDatasetInfo


class MelanoMixInfo(BaseDatasetInfo):
    def __init__(self):
        super().__init__(
            labels=[
                "akiec",
                "bcc",
                "ben_oth",
                "bkl",
                "df",
                "inf",
                "mal_oth",
                "mel",
                "nv",
                "sccka",
                "vasc"
            ]
        )