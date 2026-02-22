from abc import ABC
from typing import Optional


class BaseDatasetInfo(ABC):
    labels: list[str]
    num_classes: int

    id2label: dict[int, str]
    label2id: dict[str, int]

    description: Optional[str]

    def __init__(
        self,
        labels: list[str],
        description: Optional[str] = None
    ):
        if not labels:
            raise ValueError("labels must be a non-empty list")

        self.labels = labels
        self.num_classes = len(labels)

        # Create mappings
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.description = description

    def get_by_id(self, class_id: int) -> str:
        if class_id not in self.id2label:
            raise ValueError(f"Invalid class_id: {class_id}")
        return self.id2label[class_id]

    def get_by_label(self, label_name: str) -> int:
        if label_name not in self.label2id:
            raise ValueError(f"Invalid label_name: {label_name}")
        return self.label2id[label_name]