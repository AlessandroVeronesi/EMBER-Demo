
import torch
import torch.nn as nn
import numpy as np

########################################
### Quantization Utils

def MinMaxObserver(array):
    min = array.min()
    max = array.max()

    return min, max

def SymCalibration(array, observer, bitwidth):
    alpha, beta = observer(array)
    upbound = max(abs(alpha), abs(beta))
    if (upbound == 0):
        scale = 1
        offset = 0
    else:
        Gamma = (2**(bitwidth-1))-1
        scale = Gamma / upbound
        offset = 0

    return scale, offset

def Quantize(array, scale, offset, dtype=torch.int32):
    return torch.floor(torch.multiply(torch.subtract(array, offset), scale)).to(dtype)

def Dequantize(array, scale, offset, dtype=torch.float32):
    return torch.add(torch.divide(array.to(dtype), scale), offset)


def symQuantize(tensor, bitwidth, dtype=torch.int64):
    scale, offset = SymCalibration(tensor, MinMaxObserver, bitwidth)
    qTensor = Quantize(tensor, scale, offset, dtype)
    return qTensor, scale, offset


########################################
### Extension of Torch Layers

## Conv2d
class Conv2d(nn.Conv2d):
    """
    Conv2d with Bias layer that exploits the available nvdlasim layer
    """
    def __init__(self, nvdla, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), bias=True, profiler=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.stride     = stride
        self.padding    = padding
        self.dilation   = dilation
        self.nvdla      = nvdla
        self.bitwidth   = nvdla.bitwidth
        self.profiler   = profiler

    def __str__(self):
        superstr = super().__str__()
        return f'[nvdla]{superstr}'

    def __repr__(self):
        superstr = super().__repr__()
        return f'[nvdla]{superstr}'

    def forward(self, input):            
        ftype  = input.dtype
        qtype  = torch.int64
        nptype = np.int64

        if input.ndim > 4:
            input4d = input.reshape((-1,) + input.shape[-3:])
        else:
            input4d = input


        qFT, fScale, fOff = symQuantize(input4d.detach(), self.bitwidth, qtype)
        qWT, wScale, wOff = symQuantize(self.weight.data, self.bitwidth, qtype)

        if(self.bias is None):
            qPsums = torch.from_numpy(self.nvdla.conv(qFT.numpy().astype(nptype), qWT.numpy().astype(nptype), stride=self.stride, padding=self.padding, dilation=self.dilation, logfile=self.profiler))
        else:
            qB = Quantize(self.bias.data, (fScale*wScale), (fOff+wOff), qtype)
            qPsums = torch.from_numpy(self.nvdla.convBias(qFT.numpy().astype(nptype), qWT.numpy().astype(nptype), qB.numpy().astype(nptype), stride=self.stride, padding=self.padding, dilation=self.dilation, logfile=self.profiler))

        output4d = Dequantize(qPsums, (fScale*wScale), (fOff+wOff), dtype=ftype)

        if input.ndim > 4:
            return output4d.reshape(*input.shape[:-3], output4d.shape[-3:])
        else:
            return output4d


## Linear
class Linear(nn.Linear):
    """
    Linear with Bias layer that exploits the available nvdlasim layer
    """
    def __init__(self, nvdla, in_features, out_features, bias=True, profiler=None):
        super().__init__(in_features, out_features, bias=bias)
        self.nvdla      = nvdla
        self.bitwidth   = nvdla.bitwidth
        self.profiler   = profiler

    def __str__(self):
        superstr = super().__str__()
        return f'[nvdla]{superstr}'

    def __repr__(self):
        superstr = super().__repr__()
        return f'[nvdla]{superstr}'

    def forward(self, input):            
        ftype  = input.dtype
        qtype  = torch.int64
        nptype = np.int64

        if input.ndim > 2:
            input2d = input.reshape(-1, input.shape[-1])
        else:
            input2d = input

        qFM, fScale, fOff = symQuantize(input2d.detach(), self.bitwidth, qtype)
        qWM, wScale, wOff = symQuantize(self.weight.data, self.bitwidth, qtype)

        qFT = self.nvdla.rollback(qFM.numpy().astype(nptype))
        qWT = self.nvdla.rollback(qWM.numpy().astype(nptype))

        if(self.bias is None):
            qPT = self.nvdla.conv(qFT, qWT, stride=(1,1), padding=(0,0), dilation=(1,1), logfile=self.profiler)
        else:
            qBV = Quantize(self.bias.data, (fScale*wScale), (fOff+wOff), qtype).numpy().astype(nptype)
            qPT = self.nvdla.convBias(qFT, qWT, qBV, stride=(1,1), padding=(0,0), dilation=(1,1), logfile=self.profiler)

        qPM = torch.from_numpy(self.nvdla.unroll(qPT))

        output2d = Dequantize(qPM, (fScale*wScale), (fOff+wOff), dtype=ftype)

        if input.ndim > 2:
            return output2d.reshape(*input.shape[:-1], output2d.shape[-1])
        else:
            return output2d

## ReLU
class ReLU(nn.ReLU):
    """
    ReLU layer that exploits the available nvdlasim layer
    """
    def __init__(self, nvdla, inplace=False, profiler=None):
        super().__init__(inplace)
        self.nvdla    = nvdla
        self.bitwidth = nvdla.bitwidth
        self.profiler = profiler

    def __str__(self):
        return '[nvdla]%s' % super().__str__()

    def __repr__(self):
        return '[nvdla]%s' % super().__repr__()

    def forward(self, input):            
        ftype  = input.dtype
        qtype  = torch.int64
        nptype = np.int64

        qF, fScale, fOff = symQuantize(input.detach(), self.bitwidth, qtype)

        flatten = True if (qF.numpy().ndim == 2) else False
        if(flatten):
            qF = self.nvdla.rollback(qF)

        qP = torch.from_numpy(self.nvdla.relu(qF.numpy().astype(nptype), logfile=self.profiler))

        if(flatten):
            qP = self.nvdla.unroll(qP)

        return Dequantize(qP, fScale, fOff, dtype=ftype)