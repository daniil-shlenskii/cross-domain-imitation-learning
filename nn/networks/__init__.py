from .mlp import MLP
from .resnet import ResNet
from .utils import default_init

networks_mapping = {
    "MLP": MLP,
    "ResNet": ResNet
}
