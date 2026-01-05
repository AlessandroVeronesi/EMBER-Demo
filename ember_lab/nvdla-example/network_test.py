
import os
import sys
import argparse
import torch
import yaml
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import getCIFAR10
from utils.alexnet import AlexNet
from utils.lenet import LeNet
from utils.parser import nvdlaReplacer as converter

###################################################

if __name__ == '__main__':

    #### Args Parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize',      type=int, default=128)
    parser.add_argument('--config',         type=str, required=True,   help='NVDLA configuration')
    parser.add_argument('--profile',        action='store_true',       help='Profile the evaluated network')
    parser.add_argument('--outdir',         type=str, default='./',    help='Output directory (for profiler)')

    args = parser.parse_args()


    #### Model Factory
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    modelname = 'lenet'

    print('Building model..,')
    _, testloader, _ = getCIFAR10(datadir, 32, args.batchsize, False)
    model            = LeNet() # Random weights, just profiling
    model.to('cpu')

    print(f'-I({__file__}): Printing Loaded Net...')
    print(model)

    #### Parse and Convert
    if args.profile:
        print(f'-I({__file__}): Enabling Profiler Mode...')
        nvconverter = converter(args.config, True, args.outdir)

    else:
        nvconverter = converter(args.config)

    print(f'-I({__file__}): Replacing Net...')
    nvmodel = nvconverter.replaceAll(model)

    print(f'-I({__file__}): Printing Converted Net...')
    print(nvmodel)

    if args.profile:

        ## Profile Input
        print(f'-I({__file__}): Profiling Net Execution...')
        with torch.no_grad():
            for X, y in testloader:
                X = X.to('cpu')
                y = y.to('cpu')
                _ = nvmodel(X)
                break

    else:

        ## Base Execution
        print(f'-I({__file__}): Net CPU Execution...')
        with torch.no_grad():
            for X, y in testloader:
                X = X.to('cpu')
                y = y.to('cpu')

                base_start  = datetime.now()
                _ = model(X)
                base_stop   = datetime.now()
                break
        
        ## EMBER Execution
        print(f'-I({__file__}): Net NVDLA-EMBER Execution...')
        with torch.no_grad():
            for X, y in testloader:
                X = X.to('cpu')
                y = y.to('cpu')

                nv_start  = datetime.now()
                _ = nvmodel(X)
                nv_stop   = datetime.now()
                break
        
        ## Elapsed
        print(f'-I({__file__}): CPU Time:         {(base_stop-base_start).total_seconds()} s')
        print(f'-I({__file__}): NVDLA-EMBER Time: {(nv_stop-nv_start).total_seconds()} s')