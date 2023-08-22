import torch
import open_clip


class OpenCLIP(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""
    def __init__(self, model_name="RN50", pretrained_weights=None, requires_grad=False, show_progress=False):
        super().__init__()

        pretrained = pretrained_weights.lower()
        if pretrained in open_clip.list_pretrained_tags_by_model(model_name):
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            raise Exception(f'Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} {model_name} model.')
        
        self.model = model.eval().to(DEVICE)
        self.layer_names = ["logits_per_image"]

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
          for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        img, text = x
        img = img.to(DEVICE)
        text = self.tokenizer(text).to(DEVICE)

        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        out = (100.0 * image_features @ text_features.T)
        
        # Feel free to experiment with different layers.
        openclip_outputs = namedtuple("OpenCLIPOutputs", self.layer_names)
        out = openclip_outputs(out)
        return out