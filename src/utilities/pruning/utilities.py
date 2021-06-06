import torch
from torch import nn


@torch.no_grad()
def get_activation(preact_dict, param_name, hook_type):
    """
    Hooks used for in sensitivity schedulers (LOBSTE, Neuron-LOBSTER, SERENE).
    :param preact_dict: Dictionary in which save the parameters information.
    :param param_name: Name of the layer, used a dictionary key.
    :param hook_type: Hook type.
    :return: Returns a forward_hook if $hook_type$ is forward, else a backward_hook.
    """

    def forward_hook(model, inp, output):
        preact_dict[param_name] = output

    def backward_hook(module, grad_input, grad_output):
        preact_dict[param_name] = None
        preact_dict[param_name] = grad_output[0].detach().cpu()

    return forward_hook if hook_type == "forward" else backward_hook


@torch.no_grad()
def apply_mask_params(model, mask):
    """
    Element-wise multiplication between a tensor and the corresponding mask.
    :param mask: Dictionary containing the tensor mask at the given key.
    """
    for n_m, mo in model.named_modules():
        for n_p, p in mo.named_parameters():
            name = "{}.{}".format(n_m, n_p)
            p.mul_(mask[name])


@torch.no_grad()
def apply_mask_neurons(model, mask):
    """
    Element-wise multiplication between a tensor and the corresponding mask.
    :param mask: Dictionary containing the tensor mask at the given key.
    """
    for n_m, mo in model.named_modules():
        if isinstance(mo, (nn.modules.Linear, nn.modules.Conv2d, nn.modules.ConvTranspose2d, nn.modules.BatchNorm2d)):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)
                if len(p.shape) == 1:
                    p.mul_(mask[name])
                elif len(p.shape) == 2:
                    p.copy_(torch.einsum(
                        'ij,i->ij',
                        p,
                        mask[name]
                    ))
                elif len(p.shape) == 4:
                    if isinstance(mo, nn.modules.Conv2d):
                        p.copy_(torch.einsum(
                            'ijnm,i->ijnm',
                            p,
                            mask[name]
                        ))

                    if isinstance(mo, nn.modules.ConvTranspose2d):
                        p.copy_(torch.einsum(
                            'ijnm,j->ijnm',
                            p,
                            mask[name]
                        ))


@torch.no_grad()
def get_model_mask_neurons(model, layers):
    """
    Defines a dictionary of type {layer: tensor} containing for each layer of a model, the binary mask representing
    which neurons have a value of zero (all of its parameters are zero).
    :param model: PyTorch model.
    :param layers: Tuple of layers on which apply the threshold procedure. e.g. (nn.modules.Conv2d, nn.modules.Linear)
    :return: Mask dictionary.
    """
    mask = {}
    for n_m, mo in model.named_modules():
        if isinstance(mo, layers):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)

                if "weight" in n_p:
                    if isinstance(mo, nn.modules.Linear):
                        sum = torch.abs(p).sum(dim=1)
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    elif isinstance(mo, nn.modules.Conv2d):
                        sum = torch.abs(p).sum(dim=(1, 2, 3))
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    elif isinstance(mo, nn.modules.ConvTranspose2d):
                        sum = torch.abs(p).sum(dim=(0, 2, 3))
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    else:
                        mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))
                else:
                    mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))

    return mask


@torch.no_grad()
def get_model_mask_parameters(model, layers):
    """
    Defines a dictionary of type {layer: tensor} containing for each layer of a model, the binary mask representing
    which parameters have a value of zero.
    :param model: PyTorch model.
    :param layers: Tuple of layers on which apply the threshold procedure. e.g. (nn.modules.Conv2d, nn.modules.Linear)
    :return: Mask dictionary.
    """
    mask = {}
    for n_m, mo in model.named_modules():
        if isinstance(mo, layers):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)
                mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))

    return mask


@torch.no_grad()
def magnitude_threshold(model, layers, T, ):
    """
    Performs magnitude thresholding on a network, all the elements of the tensor below a threshold are zeroed.
    :param model: PyTorch model on which apply the thresholding, layer by layer.
    :param layers: Tuple of layers on which apply the threshold procedure. e.g. (nn.modules.Conv2d, nn.modules.Linear)
    :param T: Threhsold value.
    """
    for n_m, mo in model.named_modules():
        if isinstance(mo, layers):
            for n_p, p in mo.named_parameters():
                p.copy_(torch.where(torch.abs(p) < T, torch.zeros_like(p), p))
