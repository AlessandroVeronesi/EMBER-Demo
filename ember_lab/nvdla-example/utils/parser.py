
import os
import copy
import torch.nn as nn

from wrapper import nvdlaWrap
import nvtorch

######################################################################
## Parser Utilities

def replace_module(model, old_module, replacer, module_name = ''):
    # Recursively replace modules in the copied model
    for name, module in model.named_children():
        full_name = f'{module_name}.{name}' if module_name != '' else name
        if not hasattr(module, 'nvdla'):
            if isinstance(module, old_module):
                if hasattr(module, 'groups'):
                    if (module.groups > 1):
                        # Bypass DW layers
                        return model
                # Instantiate the replacementV module
                replacement = replacer(module, full_name)
                setattr(model, name, replacement)
            else:
                # If the module has children, apply recursively
                replace_module(module, old_module, replacer, full_name)
    
    return model  # Return the modified model copy

def replace_singleModule(model, old_module_name, supported_list, replacers_list, module_name = ''):
    # Recursively replace modules in the copied model
    for name, module in model.named_children():
        full_name = f'{module_name}.{name}' if module_name != '' else name
        if not hasattr(module, 'nvdla'):
            # if isinstance(module, old_module):
            if full_name == old_module_name:
                # Instantiate the replacement module
                for idx, supported_module in enumerate(supported_list):
                    if isinstance(module, supported_module):
                        replacer = replacers_list[idx]
                        replacement = replacer(module, full_name)
                        setattr(model, name, replacement)
            #     replacement = replacer(module, full_name)
            #     setattr(model, name, replacement)
            else:
                # If the module has children, apply recursively
                replace_singleModule(module, old_module_name, supported_list, replacers_list, full_name)
    
    return model  # Return the modified model copy

######################################################################
## NvDla Replacement Class

class nvdlaReplacer():
    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], str):
                self.nvdla   = nvdlaWrap(args[0])
                self.savedir = './'
            else:
                raise ValueError(f'Unrecognized init options\naccepted ([str] config), ([str] config, [bool] profile), ([str] savedir)\nobtained {args}')
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], bool):
                self.nvdla   = nvdlaWrap(args[0], profiler=args[1])
                self.savedir = './'
            else:
                raise ValueError(f'Unrecognized init options\naccepted ([str] config), ([str] config, [bool] profile), ([str] savedir)\nobtained {args}')
        elif len(args) == 3:
            if isinstance(args[0], str) and isinstance(args[1], bool) and isinstance(args[2], str):
                self.nvdla   = nvdlaWrap(args[0], profiler=args[1])
                self.savedir = args[2]
            else:
                raise ValueError(f'Unrecognized init options\naccepted ([str] config), ([str] config, [bool] profile), ([str] savedir)\nobtained {args}')
        else:
            raise ValueError(f'Unrecognized init options\naccepted ([str] config), ([str] config, [bool] profile), ([str] savedir)\nobtained {args}')

    def __str__(self):
        return "<nvdlaReplacer>:\nself.nvdla: %s" % self.nvdla

    ## Replacement function
    def conv(self, module, name):
        newmodule = nvtorch.Conv2d(self.nvdla,
            in_channels=module.in_channels,
            out_channels = module.out_channels,
            kernel_size = module.kernel_size,
            stride = module.stride,
            padding = module.padding,
            dilation = module.dilation,
            bias = (module.bias is not None),
            profiler=os.path.join(self.savedir, f'{name}.yaml'))
        
        newmodule.load_state_dict(module.state_dict())

        return newmodule
    
    def linear(self, module, name):
        newmodule = nvtorch.Linear(self.nvdla,
            in_features=module.in_features,
            out_features = module.out_features,
            bias = (module.bias is not None),
            profiler=os.path.join(self.savedir, f'{name}.yaml'))
        
        newmodule.load_state_dict(module.state_dict())

        return newmodule

    def relu(self, module, name):
        newmodule = nvtorch.ReLU(self.nvdla,
            profiler=os.path.join(self.savedir, f'{name}.yaml'))
        
        newmodule.load_state_dict(module.state_dict())

        return newmodule


    ## Full Net Replacements -- Internal
    def replaceConv2d(self, model):
        return replace_module(model, nn.Conv2d, self.conv)

    def replaceLinear(self, model):
        return replace_module(model, nn.Linear, self.linear)

    def replaceReLU(self, model):
        return replace_module(model, nn.ReLU, self.relu)


    ## Full Net Replacements
    def replaceAllConv2d(self, model):
        newmodel = copy.deepcopy(model)
        return self.replaceConv2d(newmodel)

    def replaceAllLinear(self, model):
        newmodel = copy.deepcopy(model)
        return self.replaceLinear(newmodel)

    def replaceAllReLU(self, model):
        newmodel = copy.deepcopy(model)
        return self.replaceReLU(newmodel)
    
    def replaceAll(self, model):
        model.to('cpu')
        newmodel = copy.deepcopy(model)
        newmodel = self.replaceConv2d(newmodel)
        newmodel = self.replaceLinear(newmodel)
        # newmodel = self.replaceReLU(newmodel)
        return newmodel
    
    def replaceList(self, model, layers):
        model.to('cpu')
        newmodel = copy.deepcopy(model)

        for layer in layers:
            newmodel = replace_singleModule(newmodel, layer, [nn.Conv2d, nn.Linear], [self.conv, self.linear])

        return newmodel