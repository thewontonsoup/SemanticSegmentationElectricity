import segmentation_models_pytorch as smp
import torch
import urllib
from torch import nn


class UNetPlusPlus(nn.Module):
    def __init__(self, input_channels, output_channels, encoder_name="resnet101",
                 encoder_weights="imagenet", decoder_channels=(256, 128, 64, 32, 16), activation=None,
                 scale_factor=50, **kwargs):
        """
        Initializes the UNet++ model with the specified input and output channels.
        The scale factor is used to pool the output of the model to the desired size.

        Input:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        encoder_name (str, optional): The name of the encoder architecture. Defaults to "resnet101".
            - List of available encoders: https://smp.readthedocs.io/en/latest/encoders.html
        encoder_weights (str, optional): The weights to use for the encoder. Defaults to "imagenet".
        decoder_channels (tuple, optional): The number of channels in the decoder blocks. Defaults to (256, 128, 64, 32, 16).
        activation (str, optional): The activation function to use. Defaults to None.
            - “sigmoid”, “softmax”, “logsoftmax”, “tanh”, “identity”, callable or None
        scale_factor (int, optional): The scale factor used to pool the output of the model. Defaults to 50.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale_factor = scale_factor
        try:
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_depth=len(decoder_channels),
                encoder_weights=encoder_weights,
                in_channels=input_channels,
                classes=output_channels,
                decoder_channels=decoder_channels,
                activation=activation
            )
        except urllib.error.URLError:
            # This is a hack to bypass the SSL error that occurs when downloading the model weights
            # This removes the SSL verification for the download
            # https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3#answer-49174340
            # https://github.com/pytorch/pytorch/issues/33288#issuecomment-1086779194
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_depth=len(decoder_channels),
                encoder_weights=encoder_weights,
                in_channels=input_channels,
                classes=output_channels,
                decoder_channels=decoder_channels,
                activation=activation
            )
        self.pool = nn.AvgPool2d(scale_factor)

    def forward(self, x):
        """
        Runs predictions on the UNet++ model followed by pooling.
        """
        # pad the input to the model from 200x200 to 224x224
        x = nn.functional.pad(x, (12, 12, 12, 12))
        pred_y = self.model(x)
        pred_y = self.pool(pred_y)
        return pred_y
