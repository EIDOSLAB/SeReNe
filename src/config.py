from torch import nn

# ArgumentParser configs
AVAILABLE_MODELS = ["lenet300", "lenet5", "alexnet", "vgg16_1", "vgg16_2", "resnet32", "resnet18", "resnet101"]
AVAILABLE_DATASETS = ["mnist", "fashion-mnist", "cifar10", "cifar100", "imagenet"]
AVAILABLE_SENSITIVITIES = ["", "serene"]
AVAILABLE_SERENE = ["full", "lower-bound", "local"]

# Logs root directory
LOGS_ROOT = "logs"

# Layers considered during regularization and pruning
LAYERS = (nn.modules.Linear, nn.modules.Conv2d)
