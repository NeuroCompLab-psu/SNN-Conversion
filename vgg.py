import torch
import torch.nn as nn

class VGG_15_avg_before_relu(nn.Module):
    def __init__(self,  dr=0.1, num_classes=1000, units=512*7*7):
        super(VGG_15_avg_before_relu, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),  # AvgPool2d,
            nn.ReLU(),

            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 1, bias=False),
            nn.AvgPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(units, 4096, bias=False),  # Linear,
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(4096, num_classes, bias=False)  # Linear,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_15_avg_before_relu(dataset='imagenet' , **kwargs):
    if dataset == 'imagenet':
        model = VGG_15_avg_before_relu(num_classes=1000, **kwargs)
    elif dataset == 'cifar100':
        model = VGG_15_avg_before_relu(num_classes=100, units=512,**kwargs)
    else:
        model = None
        raise ValueError('Unsupported Dataset!')
    return model
