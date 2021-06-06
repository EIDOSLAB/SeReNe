import torch
from torch import nn, relu
from torch.cuda.amp import autocast
from tqdm import tqdm

from .. import utilities


class NeuronLOBSTER:
    def __init__(self, model, lmbda, layers, bn_prune):
        """
        Initialize the LOBSTER regularizer.
        :param model: PyTorch model.
        :param lmbda: Lambda hyperparameter.
        :param layers: Tuple of layer on which apply the regularization e.g. (nn.modules.Conv2d, nn.modules.Linear)
        """
        self.model = model
        self.lmbda = lmbda
        self.layers = layers
        # Dict containing the preactivation of each layer as a Tensor of the layer's dimensions
        self.preactivations = {}
        self.hooks = []
        self.bn_prune = bn_prune

        self.scaler = None

        self.eps = torch.tensor([1e-10])
        self.momentum = 0.9
        self.dampening = 0
        self.buf = {}

        self.add_hooks()

    def add_hooks(self):
        # Attach to each layer a backward hook that allow us to automatically extract the preactivation
        # during the loss' backward pass
        modules = list(self.model.named_modules())

        for i, (n, mo) in enumerate(modules):
            if isinstance(mo, self.layers):
                if len(list(mo.children())) == 0 and len(list(mo.parameters())) != 0:
                    if self.bn_prune and isinstance(mo, (nn.modules.Conv2d, nn.modules.ConvTranspose2d)):
                        if i + 1 < len(modules):
                            if isinstance(modules[i + 1][1], nn.modules.BatchNorm2d):
                                continue
                    handle = mo.register_backward_hook(utilities.get_activation(self.preactivations, n, "backward"))
                    self.hooks.append(handle)

    @torch.no_grad()
    def step(self, mask_params, mask_neurons, rescaled=False):
        """
        Regularization step.
        :param masks: Dictionary of type `layer: tensor` containing, for each layer of the network a tensor
        with the same size of the layer that is element-wise multiplied to the layer.
        See `utilities.get_mask_neur` or `utilities.get_mask_par` for an example of mask construction.
        :param rescaled: If True rescale the sensitivity in [0, 1] as sensitivty /= max(sensitivity)
        """
        modules = list(self.model.named_modules())

        for i, (n_m, mo) in enumerate(modules):
            if isinstance(mo, self.layers):
                if self.bn_prune and isinstance(mo, (nn.modules.Conv2d, nn.modules.ConvTranspose2d)):
                    if i + 1 < len(modules):
                        if isinstance(modules[i + 1][1], nn.modules.BatchNorm2d):
                            continue

                for n_p, p in mo.named_parameters():

                    # Weight
                    if "weight" in n_p:
                        # Compute insensitivity only for weight params
                        # in order to avoid multiple computation of the same value
                        # Tensor [neurons, examples, etc.]
                        if self.scaler is not None:
                            preact = self.preactivations[n_m] * 1. / self.scaler.get_scale()
                        else:
                            preact = self.preactivations[n_m]

                        preact = preact.float()

                        if len(preact.shape) == 4:
                            preact = torch.mean(preact, dim=(0, 2, 3))
                        else:
                            preact = torch.mean(preact, dim=0)

                        # Momentum
                        if n_m not in self.buf:
                            buf = self.buf[n_m] = torch.clone(preact).detach()
                        else:
                            buf = self.buf[n_m]
                            buf.mul_(self.momentum).add_(preact, alpha=1. - self.dampening)

                        preact = buf
                        sensitivity = torch.abs(preact)

                        if rescaled:
                            sensitivity /= torch.max(torch.max(sensitivity), self.eps)

                        insensitivity = (1. - sensitivity).to(p.device)

                        if not rescaled:
                            insensitivity = relu(insensitivity)

                        if insensitivity.shape[0] != 1:
                            # Neuron-by-neuron (channel-by-channel) w * Ins
                            if isinstance(mo, nn.modules.Linear):
                                regu = torch.einsum(
                                    'ij,i->ij',
                                    p,
                                    insensitivity
                                )
                            elif isinstance(mo, nn.modules.Conv2d):
                                regu = torch.einsum(
                                    'ijnm,i->ijnm',
                                    p,
                                    insensitivity
                                )
                            elif isinstance(mo, nn.modules.ConvTranspose2d):
                                regu = torch.einsum(
                                    'ijnm,j->ijnm',
                                    p,
                                    insensitivity
                                )
                            else:
                                regu = torch.mul(p, insensitivity)
                        else:
                            regu = torch.mul(p, insensitivity)

                    # Bias
                    else:
                        regu = torch.mul(p, insensitivity)

                    p.add_(regu, alpha=-self.lmbda)  # w - lmbd * w * Ins

        if mask_params is not None:
            utilities.apply_mask_params(mask_params)
        elif mask_neurons is not None:
            utilities.apply_mask_neurons(self.model, mask_neurons)

    @torch.enable_grad()
    def evaluate_sensitivity(self, dataloader, loss_function, device):

        def cumulate_features(dictionary, param_name, scaler):

            def hook(module, input, output):
                # import pydevd
                # pydevd.settrace(suspend=False, trace_only_current_thread=True)

                if param_name in dictionary:
                    dictionary[param_name] += output.detach().cpu()
                else:
                    dictionary[param_name] = output.detach().cpu()

            return hook

        def cumulate_sensitivity(dictionary, param_name, scaler):

            def hook(module, grad_input, grad_output):
                # import pydevd
                # pydevd.settrace(suspend=False, trace_only_current_thread=True)

                sens = grad_output[0] * 1. / scaler.get_scale() if scaler is not None \
                    else grad_output[0]

                if len(sens.shape) == 4:
                    sens = torch.mean(sens, dim=(0, 2, 3))
                else:
                    sens = torch.mean(sens, dim=0)

                if param_name in dictionary:
                    dictionary[param_name] += sens.detach().cpu()
                else:
                    dictionary[param_name] = sens.detach().cpu()

            return hook

        sensitivity = {}
        features = {}
        handles = []

        modules = list(self.model.named_modules())

        for i, (n, mo) in enumerate(modules):
            if isinstance(mo, self.layers):
                if len(list(mo.children())) == 0 and len(list(mo.parameters())) != 0:
                    if self.bn_prune and isinstance(mo, (nn.modules.Conv2d, nn.modules.ConvTranspose2d)):
                        if i + 1 < len(modules):
                            if isinstance(modules[i + 1][1], nn.modules.BatchNorm2d):
                                continue

                    h_bw = mo.register_backward_hook(cumulate_sensitivity(sensitivity, n, self.scaler))
                    # h_fw = mo.register_forward_hook(cumulate_features(features, n, self.scaler))
                    # handles.append(h_fw)
                    handles.append(h_bw)

        self.model.eval()
        self.model.zero_grad()

        pbar = tqdm(dataloader, total=len(dataloader))
        pbar.set_description("Evaluating sensitivity and feature maps")

        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = loss_function(output, target)
            else:
                output = self.model(data)
                loss = loss_function(output, target)

            self.scaler.scale(loss).backward() if self.scaler is not None else loss.backward()

        for h in handles:
            h.remove()

        for k in sensitivity:
            # features[k] = features[k].float()
            # features[k] = torch.mean(features[k], dim=0)
            # features[k] /= len(dataloader)

            sensitivity[k] = sensitivity[k].float()
            sensitivity[k] = torch.abs(sensitivity[k])
            sensitivity[k] /= len(dataloader)
            sensitivity[k] /= torch.max(torch.max(sensitivity[k]), self.eps)

        return sensitivity, features

    def set_lambda(self, lmbda):
        self.lmbda = lmbda

    def set_scaler(self, scaler):
        self.scaler = scaler
