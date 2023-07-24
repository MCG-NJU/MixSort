from torch.hub import load_state_dict_from_url

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}


def build_deeplab(pretrained=True):
    model = deeplabv3_resnet50(
        num_classes=1,
    )

    if pretrained:
        # classification head unmatch
        model_url = model_urls['deeplabv3_resnet50_coco']
        exclude = {'classifier.4.weight', 'classifier.4.bias', 'aux_classifier.4.weight', 'aux_classifier.4.bias'}

        state_dict = load_state_dict_from_url(model_url, progress=True)
        state_dict = {k: v for k, v in state_dict.items() if k not in exclude}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("missing: {}".format(missing))
        print("unexpected: {}".format(unexpected))

    return model