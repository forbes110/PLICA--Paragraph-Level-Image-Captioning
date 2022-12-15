import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
)


class ImgCaptionModel(nn.Module):
    def __init__(self, encoder_name="google/vit-base-patch16-224-in21k", decoder_name="bert-base-uncased"):
        super().__init__()
        self.device = torch.device("cuda")
        print("Using device: ", self.device)
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_name, decoder_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_processor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k")

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, image, caption=None):
        pixel_values = self.image_processor(
            image, return_tensors="pt").pixel_values.to(self.device)

        labels = self.tokenizer(
            caption,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).input_ids.to(self.device)

        return self.model(pixel_values=pixel_values, labels=labels)

    def inference(self, image):
        pixel_values = self.image_processor(
            image, return_tensors="pt").pixel_values.to(self.device)

        generated_ids = self.model.generate(pixel_values)

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def save_model(self, model_path="model"):
        self.tokenizer.save_pretrained(model_path)
        self.model.save_pretrained(model_path)

    def load_model(self, model_path):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.image_processor = ViTFeatureExtractor.from_pretrained(model_path)

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
