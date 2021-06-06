import torch
from torch import autograd

from .. import utilities


class SERENE:
    def __init__(self, model, approach, lmbda, alpha, layers):
        self.model = model
        self.approach = approach
        self.lmbda = lmbda
        self.alpha = alpha
        self.layers = layers
        self.preactivations = {}

        if approach == "full":
            for n, mo in model.named_modules():
                if len(list(mo.children())) == 0 and len(list(mo.parameters())) != 0:
                    print("Attached hook to {}".format(n))
                    mo.register_forward_hook(utilities.get_activation(self.preactivations, n, "forward"))
        elif approach == "lower-bound":
            for n, mo in model.named_modules():
                if len(list(mo.children())) == 0 and len(list(mo.parameters())) != 0:
                    print("Attached hook to {}".format(n))
                    mo.register_backward_hook(utilities.get_activation(self.preactivations, n, "backward"))
        elif approach == "local":
            for n, mo in model.named_modules():
                if len(list(mo.children())) == 0 and len(list(mo.parameters())) != 0:
                    print("Attached hook to {}".format(n))
                    mo.register_forward_hook(utilities.get_activation(self.preactivations, n, "forward"))
        else:
            raise ValueError("Incorrect approach")

    def step(self, output, mask_params, mask_neurons):
        if self.approach == "full":
            self.step_full(output, mask_params, mask_neurons)
        elif self.approach == "lower-bound":
            self.step_lower_bound(mask_params, mask_neurons)
        elif self.approach == "local":
            self.step_local(mask_params, mask_neurons)

    def step_full(self, output, mask_params, mask_neurons):
        output = output.mean(0)
        grad = {k: 0 for k in self.preactivations}

        for y_i in output:
            for key in self.preactivations:
                grad[key] += self.alpha * torch.abs(
                    autograd.grad(y_i, self.preactivations[key], create_graph=True, retain_graph=True)[0])

        with torch.no_grad():
            for n_m, mo in self.model.named_modules():
                if isinstance(mo, self.layers):
                    for n_p, p in mo.named_parameters():
                        name = "{}.{}".format(n_m, n_p)

                        if "bias" not in n_p:
                            reshaped = False

                            grad[n_m] = grad[n_m].mean(0)

                            if len(grad[n_m].shape) > 2:
                                grad[n_m] = grad[n_m].view(grad[n_m].shape[0], -1)
                                grad[n_m] = torch.mean(grad[n_m], 1)

                            if len(p.shape) > 2:
                                original_shape = p.shape
                                target_shape = torch.Size([p.shape[0], -1])
                                p = p.view(target_shape)
                                reshaped = True

                            insensitivity = torch.nn.functional.relu(1 - grad[n_m])

                            regu = torch.einsum(
                                'ij,i->ij',
                                p,
                                insensitivity)

                            p.add_(regu, alpha=-self.lmbda)

                            if reshaped:
                                p = p.view(original_shape)

                        if mask_params is not None:
                            utilities.apply_mask_params(self.model, mask_params)
                        elif mask_neurons is not None:
                            utilities.apply_mask_neurons(self.model, mask_neurons)

    @torch.no_grad()
    def step_lower_bound(self, mask_params, mask_neurons):
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    name = "{}.{}".format(n_m, n_p)

                    if "bias" not in n_p:
                        reshaped = False

                        sensitivity = self.alpha * torch.abs(self.preactivations[n_m])
                        sensitivity = sensitivity.mean(0)

                        if len(sensitivity.shape) > 2:
                            sensitivity = sensitivity.view(sensitivity.shape[0], -1)
                            sensitivity = torch.mean(sensitivity, 1)[0]

                        if len(p.data.shape) > 2:
                            original_shape = p.data.shape
                            target_shape = torch.Size([p.data.shape[0], -1])
                            p.data = p.data.view(target_shape)
                            reshaped = True

                        insensitivity = torch.nn.functional.relu(1 - sensitivity)

                        regu = torch.einsum(
                            'ij,i->ij',
                            p.data,
                            insensitivity
                        )  # neuron-by-neuron (channel-by-channel) w * Ins

                        p.add_(regu, alpha=-self.lmbda)

                        if reshaped:
                            p.data = p.data.view(original_shape)

                        if mask_params is not None:
                            utilities.apply_mask_params(self.model, mask_params)
                        elif mask_neurons is not None:
                            utilities.apply_mask_neurons(self.model, mask_neurons)

    @torch.no_grad()
    def step_local(self, mask_params, mask_neurons):
        for n_m, mo in self.model.named_modules():
            if isinstance(mo, self.layers):
                for n_p, p in mo.named_parameters():
                    name = "{}.{}".format(n_m, n_p)

                    if "bias" not in n_p:
                        reshaped = False

                        self.preactivations[n_m] = torch.transpose(self.preactivations[n_m], 0, 1).contiguous()

                        if len(p.shape) > 2:
                            original_shape = p.shape
                            target_shape = torch.Size([p.shape[0], -1])
                            p = p.view(target_shape)
                            reshaped = True

                        if len(self.preactivations[n_m].shape) > 2:
                            self.preactivations[n_m] = self.preactivations[n_m].view(self.preactivations[n_m].shape[0],
                                                                                     -1)

                        self.preactivations[n_m] = torch.mean(self.preactivations[n_m], 1)

                        preact_tmp = torch.where(self.preactivations[n_m] > 0,
                                                 torch.zeros_like(self.preactivations[n_m]),
                                                 torch.ones_like(self.preactivations[n_m]))

                        if len(p.shape) == 2:
                            regu = torch.einsum(
                                'ij,i->ij',
                                p,
                                preact_tmp)
                        else:
                            regu = p.mul(preact_tmp)

                        p.add_(regu, alpha=-self.lmbda)

                        if reshaped:
                            p = p.view(original_shape)

                        if mask_params is not None:
                            utilities.apply_mask_params(self.model, mask_params)
                        elif mask_neurons is not None:
                            utilities.apply_mask_neurons(self.model, mask_neurons)
