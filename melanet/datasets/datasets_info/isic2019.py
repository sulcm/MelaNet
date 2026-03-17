from .base import BaseDatasetInfo


class ISIC2019Info(BaseDatasetInfo):
    def __init__(self):
        super().__init__(
            labels=[
                "nv",
                "mel",
                "bkl",
                "df",
                "sccka",
                "bcc",
                "vasc",
                "akiec"
            ]
        )