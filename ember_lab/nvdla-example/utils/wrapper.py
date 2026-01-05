
import os
import sys
import math
import tempfile
import numpy as np
import torch

## C-API for Yaml when available
import yaml

try:
    # Per PyYAML docs: use the C-accelerated variants when available.
    from yaml import CSafeLoader as YLoader, CSafeDumper as YDumper
except ImportError:
    from yaml import SafeLoader as YLoader, SafeDumper as YDumper

## Simulator Core
import pynvdla

########################################
### Helper Data Structs

PRECISION_BITWIDTH_LUT = {
    'int4':  4,
    'int8':  8,
    'int16': 16,
    'int32': 32,
    'fp8':   8,
    'fp16':  16,
    'fp32':  32,
    'bf16':  16
}

ACCUMULATOR_DTYPE_LUT = {
    'int4':  np.int32,
    'int8':  np.int32,
    'int16': np.int64,
    'int32': np.int64,
    'fp8':   np.float32,
    'fp16':  np.float32,
    'fp32':  np.float32,
    'bf16':  np.float32
}


########################################
### Wrapper Class

class nvdlaWrap:

    # Init Method
    def __init__(self, config, default_offset=0, default_rshift=0, profiler=False):
        if not os.path.isfile(config):
            raise ValueError(f'config file {config} does not exists')

        with open(config, 'r') as f:
            cfgdata = yaml.load(f, Loader=YLoader)

        self.core           = pynvdla.nvdla(config, profiler)

        self.profiler       = profiler

        self.atomicc        = cfgdata['cmac']['atomic-c']
        self.atomick        = cfgdata['cmac']['atomic-k']
        self.batchsize      = cfgdata['cmac']['batch-size']
        self.datbuf_entries = cfgdata['cbuf']['datbuf']['num-banks'] * cfgdata['cbuf']['datbuf']['bank-depth']
        self.wtbuf_entries  = cfgdata['cbuf']['wtbuf']['num-banks'] * cfgdata['cbuf']['wtbuf']['bank-depth']
        self.abuf_entries   = cfgdata['cacc']['num-banks'] * cfgdata['cacc']['bank-depth']

        self.isFloat        = ((cfgdata['dat-type'] == 'int8') or (cfgdata['dat-type'] == 'int16') or (cfgdata['dat-type'] == 'int32'))
        self.dtype          = cfgdata['dat-type']
        self.bitwidth       = self.getBitwidth(cfgdata)[0]
        self.offset         = default_offset
        self.rshift         = default_rshift

    # FIXME: Str reppresentation not implemented in pynvdla
    def __str__(self):
        print('<nvdlaWrap>:')
        print('self.profiler:', self.profiler)
        print('pynvdla.core:')
        self.core.info()
        return ""

    ## Internal Utilities
    def getBitwidth(self, config):

        datbitwidth = PRECISION_BITWIDTH_LUT[config['dat-type']]
        wtbitwidth  = PRECISION_BITWIDTH_LUT[config['wt-type']]

        return datbitwidth, wtbitwidth


    ## Output Tensor Shape
    def outShape(self, input, filter, stride=(1,1), padding=(0,0), dilation=(1,1), faults=0, seu=False):
        (B, C, H, W) = input.shape
        (K,Ck, R, S) = filter.shape

        if isinstance(padding, tuple):
            padding_h, padding_w = padding
        else:
            padding_h = padding
            padding_w = padding

        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation
            dilation_w = dilation

        if isinstance(stride, tuple):
            stride_h, stride_w = stride
        else:
            stride_h = stride
            stride_w = stride

        S_ = (S -1)*dilation_w +1
        R_ = (R -1)*dilation_h +1

        W_ = (2*padding_w +W -S_)//stride_w +1
        H_ = (2*padding_h +H -R_)//stride_h +1

        if(faults > 0) or seu:
            if(faults == 0):
                out_shape = (1, B, K, H_, W_)
            else:
                out_shape = (faults, B, K, H_, W_)
        else:
            out_shape = (B, K, H_, W_)
        return out_shape


    ## Memory Entries
    def getDatEntries(self, tensor):
        (B, C, H, W) = tensor.shape
        tilesize = H * W 
        batches = min(min(B, self.batchsize), (self.datbuf_entries // tilesize))
        tilenum  = math.ceil(C / self.atomicc)
        return tilesize, tilenum, batches

    def getWtEntries(self, tensor):
        (K, C, H, W) = tensor.shape
        tilesize = H * W * min(K, self.atomick)
        tilenum  = math.ceil(C / self.atomicc)
        return tilesize, tilenum


    # Tensor Reshape
    def unroll(self, tensor):
        if tensor.ndim == 4:
            (K, C, H, W) = tensor.shape
            return tensor.reshape((K, C*H*W))
        elif tensor.ndim == 5:
            (E, K, C, H, W) = tensor.shape
            return tensor.reshape((E, K, C*H*W))
        else:
            raise ValueError(f'Cannot unroll tensor of {tensor.ndim} dims')

    def rollback(self, matrix):
        if matrix.ndim == 2:
            (K, C) = matrix.shape
            return matrix.reshape((K,C,1,1))
        elif matrix.ndim == 3:
            (E, K, C) = matrix.shape
            return matrix.reshape((E, K,C,1,1))
        else:
            raise ValueError(f'Cannot rollback matrix of {matrix.ndim} dims')


    ## Internal pynvdla APIs
    def relu_i(self, F, fstart=0, fstop=0, fnumber=0, funit='top', logfile=''):
        return self.core.relu( F,
                                fstart=fstart, fstop=fstop, fnumber=fnumber,
                                fmodel="seu", floc=funit,
                                logfile=logfile)

    def conv_i(self, F, K, Bias, stride=(1,1), padding=(0,0), dilation=(1,1), fstart=0, fstop=0, fnumber=0, funit='top', logfile=''):
        return self.core.conv( F, K,
                                stridey=stride[0], stridex=stride[1],
                                paddingy=padding[0], paddingx=padding[1],
                                dilationy=dilation[0], dilationx=dilation[1],
                                fstart=fstart, fstop=fstop, fnumber=fnumber,
                                fmodel="seu", floc=funit,
                                logfile=logfile)

    def convBias_i(self, F, K, Bias, stride=(1,1), padding=(0,0), dilation=(1,1), fstart=0, fstop=0, fnumber=0, funit='top', logfile=''):
        return self.core.convBias( F, K, Bias,
                                    stridey=stride[0], stridex=stride[1],
                                    paddingy=padding[0], paddingx=padding[1],
                                    dilationy=dilation[0], dilationx=dilation[1],
                                    fstart=fstart, fstop=fstop, fnumber=fnumber,
                                    fmodel="seu", floc=funit,
                                    logfile=logfile)

    
    # ---------------------- SDP Tiling ---------------------------
    def sdpTile(self, coreApi, input,
                seu=False, ftile=None, fstart=0, fstop=0, fnumber=0, funit='top',
                logfile=''):
        B, C, H, W = input.shape
        res = np.empty((fnumber, B, C, H, W), dtype=input.dtype) if seu else np.empty_like(input)

        if(seu):
            res = np.empty((fnumber, B, C, H, W), dtype=input.dtype)
            for btile in range(0, B, self.batchsize):
                bend = min((btile + self.batchsize), B)
                for ctile in range(0, C, self.atomick):
                    cend = min((ctile + self.atomick), C)
                    tile_in = input[btile:bend, ctile:cend, :, :]
                    if((btile,ctile) == ftile):
                        res[:,btile:bend,ctile:cend,:,:] = coreApi(tile_in, fstart, fstop, fnumber, funit, logfile)
                    else:
                        res[:,btile:bend,ctile:cend,:,:] = coreApi(tile_in)

        elif self.profiler:

            # tilelogfile = f'temp_{os.getpid()}.yaml'
            tilelogfile = None
            count = 0
            log = {'tiles': {}}

            res = np.empty_like(input)

            # Create temp file
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete_on_close=False) as tf:
                tilelogfile = tf.name
  
                for btile in range(0, B, self.batchsize):
                    bend = min((btile + self.batchsize), B)

                    for ctile in range(0, C, self.atomick):
                        cend                           = min((ctile + self.atomick), C)
                        tile_in                        = input[btile:bend, ctile:cend, :, :]
                        res[btile:bend,ctile:cend,:,:] = coreApi(tile_in, logfile=tilelogfile)

                        # Get logfile content
                        tf.flush()
                        tf.seek(0)

                        content = tf.read()
                        tilelog = yaml.load(content, Loader=YLoader) if content else {}

                        # Record in Layer Log
                        tileid = f"tile-{count}"
                        log["tiles"][tileid] = {
                            "log": tilelog,
                            "btile": btile,
                            "ctile": ctile,
                        }

                        # Clear Temp File
                        tf.seek(0)
                        tf.truncate(0)
                        count += 1

            # Remove tempfile
            try:
                os.unlink(tilelogfile)
            except OSError:
                pass

            # Flush Complete Log
            with open(logfile, "w", encoding="utf-8") as f:
                yaml.dump(log, f, Dumper=YDumper, allow_unicode=True)

        else:
            res = np.empty_like(input)  # safe: fully overwritten
            for btile in range(0, B, self.batchsize):
                bend = min((btile + self.batchsize), B)

                for ctile in range(0, C, self.atomick):
                    cend = min((ctile + self.atomick), C)

                    tile_in = input[btile:bend, ctile:cend, :, :]
                    res[btile:bend,ctile:cend,:,:] = coreApi(tile_in)

        return res


    # ---------------------- CONV Tiling ---------------------------
    def seuConvTile_torchHelper(self, coreApi, F, K, BT, stride, padding, dilation, out_type):
        ftens_torch = torch.from_numpy(F)
        ktens_torch = torch.from_numpy(K)

        tile_out = torch.nn.functional.conv2d(
            ftens_torch, ktens_torch, bias=BT[1], stride=stride, padding=padding, dilation=dilation
        )

        return tile_out.numpy().astype(out_type, copy=False)

    def seuConvTile_pynvdlaHelper(self, coreApi, F, K, BT, stride, padding, dilation, out_type):
        return coreApi(F, K, BT[0], stride, padding, dilation).astype(out_type, copy=False)

    def seuConvTile_helperSelect(self, coreApi):

        # Torch Acceleration
        if(coreApi == self.conv_i) or (coreApi == self.convBias_i):
            return self.seuConvTile_torchHelper
        # Torch Acceleration not Available
        else:
            return self.seuConvTile_pynvdlaHelper

    ## convTile Core Routine
    def convTile(self, coreApi, Fmap, Kmap, Bias=None,
                stride=(1,1), padding=(0,0), dilation=(1,1),
                seu=False, ftile=None, fstart=0, fstop=0, fnumber=0, funit='top',
                logfile=''):

        # Prepare PSUMS
        out_shape = self.outShape(Fmap, Kmap, stride, padding, dilation, fnumber, True) if seu \
                    else self.outShape(Fmap, Kmap, stride, padding, dilation)
        acc_dtype = ACCUMULATOR_DTYPE_LUT[self.dtype]
        psums_acc = np.empty(out_shape, dtype=acc_dtype) 

        (B, _, H, W) = Fmap.shape
        (K, C, R, S) = Kmap.shape

        datTileSize, _, Bloaded = self.getDatEntries(Fmap)
        wtTileSize, _           = self.getWtEntries(Kmap)

        # Convolution Parameters Check
        if((datTileSize*Bloaded > self.datbuf_entries) or
           (wtTileSize > self.wtbuf_entries) or
           ((Bloaded * (out_shape[-2] if seu else out_shape[-2]) * (out_shape[-1] if seu else out_shape[-1])) > self.abuf_entries)):
            raise ValueError(f'Tile size exceeds memory: (dat: {datTileSize*Bloaded}/{self.datbuf_entries}; wt: {wtTileSize}/{self.wtbuf_entries}; out: {(Bloaded*(out_shape[-2])*(out_shape[-1]))}/{self.abuf_entries})')

        # cStride Calculation
        max_dat_c = (self.datbuf_entries // (datTileSize * Bloaded))
        max_wt_c  = (self.wtbuf_entries  // wtTileSize)
        cStride   = min(max_dat_c, max_wt_c) * self.atomicc

        # SEU Mode
        if(seu):
            if(psums_acc.ndim != 5):
                print("ndim != 5")
                sys.exit(1)

            # Helper Selection
            fault_free_api = self.seuConvTile_helperSelect(coreApi)

            # Cached Bias
            bias_torch_cached = torch.from_numpy(Bias) if Bias is not None else None
            B_ptr = (Bias, bias_torch_cached) if Bias is not None else (None, None)

            for btile in range(0, B, Bloaded):
                bend = min((btile + Bloaded), B)
                wrote = False

                for ctile in range(0, C, cStride):
                    cend = min((ctile + cStride), C)

                    # Input Slices
                    F = Fmap[btile:bend, ctile:cend, :, :]
                    K = Kmap[:, ctile:cend, :, :]

                    # Faulty Tile
                    if((btile,ctile) == ftile):

                        tile_out = coreApi(F, K, Bias, stride, padding, dilation,
                                           fstart, fstop, fnumber, funit, logfile)
                        tile_out = tile_out.astype(acc_dtype, copy=False)

                    # Fault-Free Tile
                    else:
                        tile_out = fault_free_api(coreApi, F, K, B_ptr, stride, padding, dilation, acc_dtype)

                    # Write result out (Assign First)
                    if not wrote:
                        psums_acc[:, btile:bend, :, :, :] = tile_out
                        wrote = True
                    else:
                        psums_acc[:, btile:bend, :, :, :] += tile_out

        # Profiler Mode
        elif(self.profiler):

            # # Tile Sizes Check
            # if((datTileSize*Bloaded > self.datbuf_entries) or (wtTileSize > self.wtbuf_entries) or ((Bloaded*psums.shape[2]*psums.shape[3]) > self.abuf_entries)):
            #     raise ValueError(f'Tile size exceeds memory: (dat: {datTileSize*Bloaded}/{self.datbuf_entries}; wt: {wtTileSize}/{self.wtbuf_entries}; out: {(Bloaded*psums.shape[2]*psums.shape[3])}/{self.abuf_entries})')

            # tilelogfile = f'temp_{os.getpid()}.yaml'
            tilelogfile = None
            log = {'tiles': {}}
            count = 0
            simtime = 0

            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete_on_close=False) as tf:
                tilelogfile = tf.name

                for btile in range(0, B, Bloaded):
                    bend = min((btile + Bloaded), B)
                    wrote = False  # item 6  # PATCH

                    for ctile in range(0, C, cStride):
                        cend = min((ctile + cStride), C)
                        F = Fmap[btile:bend, ctile:cend, :, :]
                        K = Kmap[:, ctile:cend, :, :]

                        tile_out = coreApi(F, K, Bias, stride, padding, dilation, logfile=tilelogfile).astype(acc_dtype, copy=False)

                        if not wrote:
                            psums_acc[btile:bend, :, :, :] = tile_out
                            wrote = True
                        else:
                            psums_acc[btile:bend, :, :, :] += tile_out

                        # Get logfile content
                        tf.flush()
                        tf.seek(0)

                        content = tf.read()
                        tilelog = yaml.load(content, Loader=YLoader) if content else {}

                        # Record in Layer Log
                        tileid = f'tile-{count}'
                        log['tiles'][tileid] = {
                            'tileid': count,
                            'c-tile': ctile,
                            'b-tile': btile,
                            'log':    tilelog
                            }

                        # Clear Temp File
                        tf.seek(0)
                        tf.truncate(0)
                        simtime += tilelog['total-simtime']
                        count += 1

            # Record total layer duration
            log['total-layertime'] = simtime

            # Remove tempfile
            try:
                os.unlink(tilelogfile)
            except OSError:
                pass

            # Flush Complete Log
            with open(logfile, "w", encoding="utf-8") as f:
                yaml.dump(log, f, Dumper=YDumper, allow_unicode=True)


        # Normal Mode
        else:
            for btile in range(0, B, Bloaded):
                bend = min((btile + Bloaded), B)
                wrote = False  # item 6  # PATCH

                for ctile in range(0, C, cStride):
                    cend = min((ctile + cStride), C)
                    F = Fmap[btile:bend, ctile:cend, :, :]
                    K = Kmap[:, ctile:cend, :, :]

                    tile_out = coreApi(F, K, Bias, stride, padding, dilation).astype(acc_dtype, copy=False)

                    if not wrote:
                        psums_acc[btile:bend, :, :, :] = tile_out
                        wrote = True
                    else:
                        psums_acc[btile:bend, :, :, :] += tile_out

        return psums_acc


    ## APIs
    def relu(self, Fmap, seu=False, ftile=None, fstart=0, fstop=0, fnumber=0, funit='top', logfile=''):
        return self.sdpTile(self.relu_i, Fmap, seu, ftile, fstart, fstop, fnumber, funit, logfile)

    def conv(self, Fmap, Kmap, stride=(1,1), padding=(0,0), dilation=(1,1), seu=False, ftile=None, fstart=0, fstop=0, fnumber=0, funit='top', logfile=''):
        return self.convTile(self.conv_i, Fmap, Kmap, None, stride, padding, dilation, seu, ftile, fstart, fstop, fnumber, funit, logfile)

    def convBias(self, Fmap, Kmap, Bias, stride=(1,1), padding=(0,0), dilation=(1,1), seu=False, ftile=None, fstart=0, fstop=0, fnumber=0, funit='top', logfile=''):
        return self.convTile(self.convBias_i, Fmap, Kmap, Bias, stride, padding, dilation, seu, ftile, fstart, fstop, fnumber, funit, logfile)
