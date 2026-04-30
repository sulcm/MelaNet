"""
MILK10k:
label2id: {
    "akiec": "0",
    "bcc": "1",
    "ben_oth": "2",
    "bkl": "3",
    "df": "4",
    "inf": "5",
    "mal_oth": "6",
    "mel": "7",
    "nv": "8",
    "sccka": "9",
    "vasc": "10"
}

---

HAM10000:
label2id: {
    'mel': 0,
    'nv': 1,
    'bcc': 2,
    'akiec': 3,
    'bkl': 4,
    'df': 5,
    'vasc': 6
}
"""


import numpy as np


MILK10k2HAM10000_MAPPING = {
    7: 0, # mel: MILK 7 -> HAM 0
    8: 1, # nv: MILK 8 -> HAM 1
    1: 2, # bcc: MILK 1 -> HAM 2
    0: 3, # akiec: MILK 0 -> HAM 3
    3: 4, # bkl: MILK 3 -> HAM 4
    4: 5, # df: MILK 4 -> HAM 5
    10: 6 # vasc: MILK 10 -> HAM 6
}
MILK10k2HAM10000_LABELS = list(MILK10k2HAM10000_MAPPING.keys()) # IDs that select from MILK10k model preds relevant for HAM10000


def convert_milk10k2ham10000_preds(preds: np.ndarray) -> np.ndarray:
    if preds.ndim == 1:
        return preds[MILK10k2HAM10000_LABELS]
    elif preds.ndim == 2:
        return preds[:, MILK10k2HAM10000_LABELS]
    else:
        raise ValueError(f"Predictions must have 1 or 2 dims but got {preds.ndim}")