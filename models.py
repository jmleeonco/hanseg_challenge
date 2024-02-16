""" Test new model's input and output here
Activation should be none
Input : [batch, chan, width, height] , ex [16, 1, 512, 512]
Output : [batch, class(chan), width, height] , ex [16, 2, 512, 512]
 """
import segmentation_models_pytorch as smp
import torch 
import torch.nn as nn



class Unet10(nn.Module):
    def __init__(self, ):
        super(Unet10, self).__init__()
        self.model = smp.FPN(
            encoder_name="mit_b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=31,                      # model output channels (number of classes in your dataset)
        )
    def forward(self, x):
        return self.model(x)

models_dict = {
    "Unet10" : Unet10,
}

