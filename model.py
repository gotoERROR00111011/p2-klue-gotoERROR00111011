import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x



# Custom Model Template
class MaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        #from torchvision.models import wide_resnet50_2
        #self.net = wide_resnet50_2(pretrained=True)
        from torchvision.models import vgg19_bn
        self.net = vgg19_bn(pretrained=True)
        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return x
        



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# Custom Model Template
class FaceNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained='vggface2')
        self.net.last_bn = Identity()

        #for i, param in enumerate(self.net.parameters()):
        #    param.requires_grad = False

        self.net.last_linear = nn.Sequential(
            # more layer
            # more featrue
            nn.Linear(1792, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


# Custom Model Template
class EffNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.model.last_bn = Identity()

        #for i, param in enumerate(self.net.parameters()):
        #    param.requires_grad = False

        self.model._fc = nn.Sequential(
            nn.Linear(2560, num_classes),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x


class AgeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained='vggface2')
        self.net.last_bn = Identity()

        #for i, param in enumerate(self.net.parameters()):
        #    param.requires_grad = False

        self.net.last_linear = nn.Sequential(
            # more layer
            # more featrue
            nn.Linear(1792, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


class GenderNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained='vggface2')
        self.net.last_bn = Identity()

        #for i, param in enumerate(self.net.parameters()):
        #    param.requires_grad = False

        self.net.last_linear = nn.Sequential(
            # more layer
            # more featrue
            nn.Linear(1792, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


class MaskNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained='vggface2')
        self.net.last_bn = Identity()

        #for i, param in enumerate(self.net.parameters()):
        #    param.requires_grad = False

        self.net.last_linear = nn.Sequential(
            # more layer
            # more featrue
            nn.Linear(1792, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


