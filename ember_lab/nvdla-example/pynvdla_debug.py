
import os
import sys
import argparse
import yaml

import numpy as np
import torch
from random import randrange, randint
from datetime import datetime

import pynvdla

##############################

MAXHEIGHT    = 10
MAXWIDTH     = 10
MAXWHEIGHT   = 7
MAXWWIDTH    = 7
MAXSTRIDE    = 5
MAXPADDING   = 5
MAXDILATION  = 5

##############################

# def dilation_workaround(tensor, dilation):
#     K, C, H, W = tensor.shape
#     H_ = (H-1)*dilation[0] +1
#     W_ = (W-1)*dilation[1] +1
#     dTensor = np.zeros((K,C,H_,W_), dtype=tensor.dtype)
#     dTensor[:,:,::dilation[0],::dilation[1]] = tensor
#     return dTensor

##############################

units=[
    'top',
    'top.conv',
    'top.conv.cdma',
    'top.conv.cbuf',
    'top.conv.cbuf.datbuf',
    'top.conv.cbuf.wtbuf',
    'top.conv.csb',
    'top.conv.csc',
    'top.conv.dl',
    'top.conv.wl',
    'top.conv.cmac',
    'top.conv.cacc',
    'top.conv.dbuf',
    'top.sdp',
    'top.sdp.csb',
    'top.sdp.core'
]

layers=[
    'conv',
    'relu',
    'convbias'
]

##############################

def randval(input):
    return randrange(input) if (input > 0) else 0

def isFloat(configfile):
    with open(configfile, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    if((config['dat-type'] == 'int4') or (config['dat-type'] == 'int8') or (config['dat-type'] == 'int16') or (config['dat-type'] == 'int32')):
        return False
    else:
        return True

def cfgCheck(configfile):
    with open(configfile, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    
    if(config['dat-type'] == config['wt-type']):
        return True
    else:
        return False

def filterArray(batch, filter):
    matches = np.all(batch == filter, axis=tuple(range(1, batch.ndim)))
    return batch[~matches]


##############################

def relu(ftens, ktens=None, bias=None, stride=0, padding=0, dilation=0):
    if((ftens.ndim == 4) or (ftens.ndim == 2)):
        ftens_torch = torch.from_numpy(ftens)
        out_tensor_torch = torch.nn.functional.relu(ftens_torch)
        out_tensor = out_tensor_torch.numpy()
    else:
        raise Exception('Invalid input dimensions')
    
    return out_tensor

def conv(ftens, ktens, bias, stride, padding, dilation):
    if((ftens.ndim != 4) or (ktens.ndim != 4)):
        raise Exception('Invalid input dimensions')
    
    B, C, H, W = ftens.shape
    K,C_, R, S = ktens.shape

    if(C != C_):
        raise Exception('Invalid input dimensions')

    ftens_torch = torch.from_numpy(ftens)
    ktens_torch = torch.from_numpy(ktens)

    otens_torch = torch.nn.functional.conv2d(ftens_torch, ktens_torch, stride=stride, padding=padding, dilation=dilation)

    return otens_torch.numpy()

def convBias(ftens, ktens, bias, stride, padding, dilation):
    if((ftens.ndim != 4) or (ktens.ndim != 4)):
        raise Exception('Invalid input dimensions')
    
    B, C, H, W = ftens.shape
    K,C_, R, S = ktens.shape

    if(C != C_):
        raise Exception('Invalid input dimensions')

    ftens_torch = torch.from_numpy(ftens)
    ktens_torch = torch.from_numpy(ktens)
    bias_torch  = torch.from_numpy(bias)

    otens_torch = torch.nn.functional.conv2d(ftens_torch, ktens_torch, bias=bias_torch, stride=stride, padding=padding, dilation=dilation)

    return otens_torch.numpy()

##############################

def dutRelu(nvdla, ftens, ktens=None, bias=None, stride=0, padding=0, dilation=0, 
            funit='top', fmodel='seu', fstart=0, fstop=0, fnumber=0,
            logfile=''):
    return nvdla.relu(ftens,
                      floc=funit, fmodel=fmodel, fstart=fstart, fstop=fstop, fnumber=fnumber,
                      logfile=logfile)

def dutConv(nvdla, ftens, ktens, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1),
            funit='top', fmodel='seu', fstart=0, fstop=0, fnumber=0,
            logfile=''):
    return nvdla.conv(ftens, ktens, 
                      stridex=stride[1], stridey=stride[0],
                      paddingx=padding[1], paddingy=padding[0],
                      dilationx=dilation[1], dilationy=dilation[0],
                      floc=funit, fmodel=fmodel, fstart=fstart, fstop=fstop, fnumber=fnumber,
                      logfile=logfile)

def dutConvBias(nvdla, ftens, ktens, bias, stride=(1,1), padding=(0,0), dilation=(1,1),
            funit='top', fmodel='seu', fstart=0, fstop=0, fnumber=0,
            logfile=''):
    return nvdla.convBias(  ftens, ktens, bias,
                            stridex=stride[1], stridey=stride[0],
                            paddingx=padding[1], paddingy=padding[0],
                            dilationx=dilation[1], dilationy=dilation[0],
                            floc=funit, fmodel=fmodel, fstart=fstart, fstop=fstop, fnumber=fnumber,
                            logfile=logfile)


##############################

def testroutine(nvdla, dbgcfg, layerid, dutlayer, dbglayer, testnum, funit, fstart, fstop, fnumber, seu=False, verbose=False):

    MAXB = -1
    MAXC = -1
    MAXK = -1
    with open(dbgcfg, 'r') as f:
        cfgdict = yaml.load(f, yaml.SafeLoader)
        MAXB    =      cfgdict['cmac']['batch-size']
        MAXC    = 2 *  cfgdict['cmac']['atomic-c']
        MAXK    = 32 * cfgdict['cmac']['atomic-k']

    for testit in range(testnum):

        input_gen_exit = False
        while(not input_gen_exit):
            stride   = ((randval(MAXSTRIDE-1)   +1), (randval(MAXSTRIDE-1)   +1))
            padding  = ((randval(MAXPADDING)),       (randval(MAXPADDING))      )
            dilation = ((randval(MAXDILATION-1) +1), (randval(MAXDILATION-1) +1))

            B = randval(MAXB-1) +1
            K = randval(MAXK-1) +1
            C = (randval(MAXC-1) +1) if (layerid != 'relu') else K
            R = randval(MAXWHEIGHT-1) +1
            S = randval(MAXWWIDTH-1)  +1

            R_ = (R-1)*dilation[0] +1
            S_ = (S-1)*dilation[1] +1

            H_ = randval(MAXHEIGHT) +1
            W_ = randval(MAXWIDTH) +1

            H = (H_-1)*stride[0] -2*padding[0] +R_
            W = (W_-1)*stride[1] -2*padding[1] +S_

            if((H_>0) and (W_>0) and (H>0) and (W>0) and (R_>0) and (S_>0)):
                input_gen_exit = True

        # # Static Benchmark #1
        # K = MAXK
        # R = 5
        # S = 3

        # B = MAXB
        # C = 32
        # H = 9
        # W = 6

        # stride   = (1,3)
        # padding  = (1,0)
        # dilation = (1,1)

        # # Static Benchmark #2
        # K = 191
        # R = 2
        # S = 3

        # B = 5
        # C = 8
        # H = 15
        # W = 18

        # stride   = (1,1)
        # padding  = (0,0)
        # dilation = (1,1)

        # # Static Benchmark #3
        # K = MAXK
        # R = 2
        # S = 2

        # B = MAXB
        # C = MAXC
        # H = 7
        # W = 7

        # stride   = (1,1)
        # padding  = (0,0)
        # dilation = (2,2)

        if(isFloat(dbgcfg)):
            Fmap = np.random.uniform(-10,10, (B,C,H,W)).astype(np.float32)
            Kmap = np.random.uniform(-10,10, (K,C,R,S)).astype(np.float32)
            Bias = np.random.uniform(-10,10, K).astype(np.float32)
        else:
            Fmap = np.random.randint(-10,10, (B,C,H,W), dtype=np.int64)
            Kmap = np.random.randint(-10,10, (K,C,R,S), dtype=np.int64)
            Bias = np.random.randint(-10,10, K, dtype=np.int64)

        print('-I: Running Test with:')
        print(f'-I: Layer    = {layerid}')
        print(f'-I: Fmap     = {Fmap.shape}')
        print(f'-I: Kmap     = {Kmap.shape}')
        print(f'-I: Bias     = {Bias.shape}')
        print(f'-I: Stride   = {stride}')
        print(f'-I: Padding  = {padding}')
        print(f'-I: Dilation = {dilation}')

        # print(input("Waiting user input... "))

        if verbose:
            print('-I: inputs tensor is:')
            print(Fmap)
            if (layerid != 'relu'):
                print('-I: weigths tensor is:')
                print(Kmap)
                print('-I: Bias array is:')
                print(Bias)

        print(f'-I: debug {layerid}')
        golden = dbglayer(Fmap, Kmap, Bias, stride, padding, dilation)

        # # Dilation Workaround
        # dummy_dilation = (1,1)
        # dKmap = dilation_workaround(Kmap, dilation)


        if not seu:
            print(f'-I: dut {layerid}')
            start  = datetime.now()
            dut    = dutlayer(nvdla, Fmap, Kmap, Bias, stride, padding, dilation, logfile=f'{layerid}{testit}.yaml')
            # dut    = dutlayer(nvdla, Fmap, dKmap, Bias, stride, padding, dummy_dilation, logfile=f'{layerid}{testit}.yaml')
            stop   = datetime.now()

            if((golden.shape == dut.shape) and np.allclose(golden, dut, rtol=args.threshold)):
                print(f'-I: {layerid} 4D test {testit} PASSED (elapsed: {(stop-start).total_seconds()*1000} ms)')
                if verbose:
                    print('-I: output tensor is:')
                    print(dut)
            else:
                if(golden.shape != dut.shape):
                    print(f'-E: Expected tensor of shape {golden.shape}, but received of shape {dut.shape}')
                    print(f'-E: {layerid} 4D test {testit} FAILED')
                    sys.exit('-E: DEBUG  FAILED')
                else:
                    (B,C,H,W) = golden.shape
                    for b in range(B):
                        for c in range(C):
                            if not ((golden[b][c].shape == dut[b][c].shape) and np.allclose(golden[b][c], dut[b][c], rtol=args.threshold)):
                                print(f'-E: First error in sample {b} at channel {c}')
                                print('--------------------------------------------------')
                                print('Golden:')
                                print(golden[0][0])
                                print('...')
                                print(golden[b][c])
                                print('--------------------------------------------------')
                                print('DUT')
                                print(dut[0][0])
                                print('...')
                                print(dut[b][c])
                                print('--------------------------------------------------')
                                print(f'Features Tensor Parameters: {Fmap.shape}')
                                print(f'Weights Tensor Parameters:  {Kmap.shape}')
                                print(f'Golden Tensor Parameters:   {golden.shape}')
                                print(f'DUT Tensor Parameters:      {dut.shape}')
                                print('--------------------------------------------------')
                                print(f'-E: {layerid} 4D test {testit} FAILED')
                                sys.exit('-E: DEBUG  FAILED')
        else:
            print(f'-I: Injecting location: {funit} with {nvdla.get_regs(funit)} locations...')

            print(f'-I: dut {layerid} seu')
            start  = datetime.now()
            dut    = dutlayer(nvdla, Fmap, Kmap, Bias, stride, padding, dilation, funit, 'seu', fstart, fstop, fnumber, logfile=f'seu_{layerid}{testit}.csv')
            stop   = datetime.now()

            if(dut[0].shape != golden.shape):
                print(f'-E: Expected tensors of shape {golden.shape}, but received of shape {dut[0].shape}')
                print(f'-E: {layerid} 4D test {testit} FAILED')
                sys.exit('-E: DEBUG  FAILED')
            
            if(dut.shape[0] != fnumber):
                print(f'-E: Expected tensors with {fnumber} outputs, but received of {dut.shape[0]} outputs')
                print(f'-E: {layerid} 4D test {testit} FAILED')
                sys.exit('-E: DEBUG  FAILED')

            dut = filterArray(dut, golden)
            errors = dut.shape[0]
            print(f'-I(test {testit}): injected {errors} errors over {args.fnumber} runs')

            numErrors = np.sum(dut, axis=tuple(range(1, golden.ndim)))
            print(f'-I(test {testit}): printing errors:', numErrors)

##############################

parser = argparse.ArgumentParser()

parser.add_argument('--layer', type=str, default='conv', choices=layers, help='Tested Layer')
parser.add_argument('--testnum', type=int, default=1, help='Set number of tests')
parser.add_argument('--threshold', type=float, default=1e-1, help='Set maximum error tolerance in output check')
parser.add_argument('--verbose', action='store_true', help='Verbose Testing')
parser.add_argument('--config', help='Config identifier file')
parser.add_argument('--profiler', action='store_true', help='Test profilation options')
parser.add_argument('--saf', action='store_true', help='Test stuck-at fault options')
parser.add_argument('--seu', action='store_true', help='Test single-event upset options')
parser.add_argument('--fnumber', type=int, default=1, help='Number of injected fault')
parser.add_argument('--funit', type=str, default='top', choices=units, help='Injected fault unit')
parser.add_argument('--fstart', type=int, default=0, help='Start of injection window')
parser.add_argument('--fstop', type=int, default=0, help='End of injection window')

args = parser.parse_args()

if (args.saf and args.seu):
    print('Cannot test seu and saf at the same time')
    parser.print_help()
    sys.exit(2)

##############################

pid = os.getpid()

dbgcfg = os.path.join(os.path.dirname(__file__), '..', 'specs', 'debug16_int8.yaml')

if args.config is not None:
    dbgcfg = args.config

if not os.path.isfile(dbgcfg):
    print(f'Configuration File {dbgcfg} does not exists')
    sys.exit('-E: DEBUG  FAILED')

if not cfgCheck(dbgcfg):
    print(f'Configuration File {dbgcfg} has unmatching wt and dat data types (feature not supported)')
    sys.exit('-E: DEBUG  FAILED')

print(f'\n-I: NVDLASIM Debug')
print(f'-I: PID = {pid}')
print(f'-I: Debug Config = {dbgcfg}')
print(f'-I: Running {args.testnum} tests...')

##############################

if(args.profiler):
    nvdla = pynvdla.nvdla(dbgcfg, True)
else:
    nvdla = pynvdla.nvdla(dbgcfg)

if args.verbose:
    print('-I: Debugging configuration:')
    nvdla.info() # FIXME

nvdla.info()

##############################

nvdla.clear_faults()

if args.saf:
    sa0num = 0
    sa1num = 0
    for i in range(args.fnumber):
        if randint(0, 1) == 0:
            sa0num += 1
        else:
            sa1num += 1
    nvdla.inject(fault_loc=args.funit, fault_model='sa0', fault_number=sa0num)
    nvdla.inject(fault_loc=args.funit, fault_model='sa1', fault_number=sa1num)

##############################

testnum = 1 if args.seu else args.testnum

if args.layer == 'relu':
    testroutine(nvdla, dbgcfg, args.layer, dutRelu, relu, testnum, args.funit, args.fstart, args.fstop, args.fnumber, args.seu, args.verbose)

elif args.layer == 'conv':
    testroutine(nvdla, dbgcfg, args.layer, dutConv, conv, testnum, args.funit, args.fstart, args.fstop, args.fnumber, args.seu, args.verbose)

elif args.layer == 'convbias':
    testroutine(nvdla, dbgcfg, args.layer, dutConvBias, convBias, testnum, args.funit, args.fstart, args.fstop, args.fnumber, args.seu, args.verbose)
else:
    print(f'Unsupported Layer: {args.layer}')
    sys.exit(2)

##############################

print('-I: ALL TESTS PASSED')
