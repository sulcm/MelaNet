import torch

from transformers import AutoImageProcessor, AutoModelForImageClassification


class MelaNet():
    def __init__(self, model_name: str, device: str = "cuda"):
        if device != "cpu":
            device = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        self.model = AutoModelForImageClassification.from_pretrained(model_name, device_map=self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, image, return_logits: bool = False):
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)

        logits = outputs.logits
        if return_logits:
            return logits.cpu().detach().numpy()
        else:
            predicted_class_idx = logits.argmax(-1).cpu().detach().numpy()
            return predicted_class_idx

    def __call__(self, image, return_logits: bool = False):
        return self.forward(
            image=image,
            return_logits=return_logits
        )