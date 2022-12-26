import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
    GPT2TokenizerFast
)


class ImgCapModel(nn.Module):
    '''
        Main model:
            Encoder \
                "google/vit-base-patch16-224-in21k" \
                "facebook/convnext-xlarge-384-22k-1k" \

            Decoder \
                "gpt2" \
                "bert-base-uncased" \

            Combined \
                "nlpconnect/vit-gpt2-image-captioning"
    '''

    def __init__(
        self,
        use_pretrain_imgcap=False,
        encoder_name="google/vit-base-patch16-224-in21k",
        decoder_name="gpt2",
    ):

        super().__init__()

        # check device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: ", self.device)

        self.use_pretrain_imgcap = use_pretrain_imgcap
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        if use_pretrain_imgcap is False:
            " sub-tools "
            # visual extractor for encoder
            self.image_processor = ViTFeatureExtractor.from_pretrained(
                encoder_name
            )
            " main model: with config, pretrained encoder, decoder "
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                encoder_pretrained_model_name_or_path=encoder_name,
                decoder_pretrained_model_name_or_path=decoder_name
            )

            # set tokenizer for decoder
            if self.decoder_name == "gpt2":
                self.tokenizer = GPT2TokenizerFast.from_pretrained(decoder_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
            else:
                self.tokenizer = BertTokenizer.from_pretrained(decoder_name)
                self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            print("Using pretrain img-caption model...")
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning")
            self.image_processor = ViTFeatureExtractor.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning")
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning")

        self.model.to(self.device)

    def forward(self, image, caption=None):

        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device)

        labels = self.tokenizer(
            caption,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).input_ids.to(self.device)

        return self.model(pixel_values=pixel_values, labels=labels)

    def inference(self, image, gen_kwargs):

        pixel_values = self.image_processor(
            image, return_tensors="pt").pixel_values.to(self.device)

        generated_ids = self.model.generate(pixel_values, **gen_kwargs)

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)

        generated_text = [text.replace("\n", "") for text in generated_text]

        return generated_text

    def save_model(self, model_path="model"):
        self.tokenizer.save_pretrained(model_path)
        self.image_processor.save_pretrained(model_path)
        self.model.save_pretrained(model_path)

    def load_model(self, model_path):
        if self.decoder_name == "gpt2":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.image_processor = ViTFeatureExtractor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
