import math
import collections
import logging
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch
from . import serialmodule

def add_args(parser):
    parser.add_argument("--squeezenet_in_channels",type=int, default=1)
    parser.add_argument("--squeezenet_base",type=int, default=128)
    parser.add_argument("--squeezenet_incr",type=int, default=128)
    parser.add_argument("--squeezenet_multiplicative_incr", type=int, default=2)
    parser.add_argument("--squeezenet_prop3",type=float, default=0.5)
    parser.add_argument("--squeezenet_freq",type=int, default=2)
    parser.add_argument("--squeezenet_sr",type=float, default=0.125)
    parser.add_argument("--squeezenet_out_dim",type=int)

    parser.add_argument("--squeezenet_mode",type=str, choices=["resfire","wide_resfire","dense_fire","normal"], default="normal")

    parser.add_argument("--squeezenet_dropout_rate",type=float)
    parser.add_argument("--fire_skip_mode", type=str, choices=["simple", "none"], default= "none")
    parser.add_argument("--squeezenet_pool_interval_mode",type=str, choices=["add","multiply"], default="add")
    parser.add_argument("--squeezenet_pool_interval",type=int, default=4)
    parser.add_argument("--squeezenet_num_fires", type=int, default=8)
    parser.add_argument("--squeezenet_conv1_stride", type=int, default=2)
    parser.add_argument("--squeezenet_conv1_size",type=int, default=7)
    parser.add_argument("--squeezenet_num_conv1_filters", type=int, default=96) 
    parser.add_argument("--squeezenet_pooling_count_offset", type=int, default=2) #should be greater than 0, otherwise you get a pool in the first layer
    parser.add_argument("--squeezenet_dense_k",type=int, default=12)
    parser.add_argument("--squeezenet_dense_fire_depths",type=str, default="default, shallow")
    parser.add_argument("--squeezenet_dense_fire_compress_level", type=float, default=0.5 )
    parser.add_argument("--squeezenet_use_excitation",   action="store_true")
    parser.add_argument("--squeezenet_excitation_r", type=int, default=16 )
    parser.add_argument("--squeezenet_local_dropout_rate", type=int, default=0 )
    


    


FireConfig=collections.namedtuple("FireConfig","in_channels,num_squeeze, num_expand1, num_expand3, skip")
class Fire(serialmodule.SerializableModule):
    @staticmethod 
    def from_configure(configure):
        return Fire(in_channels= configure.in_channels, num_squeeze= configure.num_squeeze,num_expand1= configure.num_expand1,num_expand3= configure.num_expand3, skip=configure.skip)


    def __init__(self, in_channels, num_squeeze, num_expand1, num_expand3, skip):
      super().__init__()
      self.squeezeconv = nn.Conv2d(in_channels, num_squeeze, (1,1))
      self.expand1conv = nn.Conv2d(num_squeeze, num_expand1, (1,1))
      self.expand3conv = nn.Conv2d(num_squeeze, num_expand3, (3,3), padding=(1,1))
      self.skip = skip
      if skip:
          assert(num_expand1+ num_expand3 == in_channels)

    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by width
        '''
        out = F.leaky_relu(self.squeezeconv(x)) # batchsize by num_squeeze by height by width
        out = [F.leaky_relu(self.expand1conv(out)), F.leaky_relu(self.expand3conv(out)) ]  #batchsize by num expand1 by height by width and batchsize by num expand1 by height by width
        out = torch.cat(out,dim=1) #batchsize by num_expand1 +num_expand 3
        if self.skip:
            out=out+x
        return out

class ResFire(serialmodule.SerializableModule):
    @staticmethod 
    def from_configure(configure):
        return ResFire(in_channels= configure.in_channels, num_squeeze= configure.num_squeeze,num_expand1= configure.num_expand1,num_expand3= configure.num_expand3, skip=configure.skip)


    def __init__(self, in_channels, num_squeeze, num_expand1, num_expand3, skip):
      super().__init__()
      self.bn1 = torch.nn.BatchNorm2d(in_channels)
      self.squeezeconv = nn.Conv2d(in_channels, num_squeeze, (1,1))
      self.bn2 = torch.nn.BatchNorm2d(num_squeeze)
      self.expand1conv = nn.Conv2d(num_squeeze, num_expand1, (1,1))
      self.expand3conv = nn.Conv2d(num_squeeze, num_expand3, (3,3), padding=(1,1))
      self.skip=skip
      if skip:
          assert(num_expand1+ num_expand3 == in_channels)
    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by width
        '''
        out = self.bn1(x)
        out = F.leaky_relu(out)
        out = self.squeezeconv(out)
        out = self.bn2(out)
        out = F.leaky_relu(out)
        out= torch.cat( [self.expand1conv(out), self.expand3conv(out)], dim=1  ) 
        if self.skip:
            out=out+x
        return out

class WideResFire(serialmodule.SerializableModule):
    '''
    Like above, but the squeeze layer has 3 by 3 convs
    '''
    @staticmethod 
    def from_configure(configure):
        return WideResFire(in_channels= configure.in_channels, num_squeeze= configure.num_squeeze,num_expand1= configure.num_expand1,num_expand3= configure.num_expand3, skip=configure.skip, local_dropout_rate=0)


    def __init__(self, in_channels, num_squeeze, num_expand1, num_expand3, skip):
      super().__init__()
      self.bn1 = torch.nn.BatchNorm2d(in_channels)
      self.squeezeconv = nn.Conv2d(in_channels, num_squeeze, (3,3), padding=(1,1))
      self.bn2 = torch.nn.BatchNorm2d(num_squeeze)
      self.expand1conv = nn.Conv2d(num_squeeze, num_expand1, (1,1))
      self.expand3conv = nn.Conv2d(num_squeeze, num_expand3, (3,3), padding=(1,1))
      self.skip=skip
      if skip:
          assert(num_expand1+ num_expand3 == in_channels)
      self.local_dropout_rate=local_dropout_rate
    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by width
        '''
        out = self.bn1(x)
        out = F.leaky_relu(out)
        out = self.squeezeconv(out)
        out = F.dropout(out, p=self.local_dropout_rate)
        out = self.bn2(out)
        out = F.leaky_relu(out)
        out= torch.cat( [self.expand1conv(out), self.expand3conv(out)], dim=1  ) 
        if self.skip:
            out=out+x
        return out


class FireSkipMode(Enum):
    NONE=0
    SIMPLE=1


class ExcitationFire(serialmodule.SerializableModule):
    def __init__(self,fire_to_wrap, in_channels, out_channels, r, skip):
        super().__init__()
        compressed_dim=max(1,math.floor( in_channels/r  ))
        self.compress=nn.Linear(in_channels, compressed_dim  )
        self.wrapped=fire_to_wrap
        self.expand=nn.Linear(compressed_dim, out_channels)
        self.skip=skip
        if skip:
            logging.info("Creating ExcitationFire with skip layer")
    def forward(self, x):
        z=torch.mean(x,3)
        z=torch.mean(z,2)
        z=self.compress(z)
        z=F.leaky_relu(z)
        z=self.expand(z)
        z=F.sigmoid(z)
        z=torch.unsqueeze(z,2)
        z=torch.unsqueeze(z,3)
        result=z*self.wrapped(x)
        if self.skip:
            result=result+x
        return result

class DenseFire(serialmodule.SerializableModule):
    def __init__(self, k0, num_subunits, k, prop3):
        super().__init__()
        self.subunit_dict=collections.OrderedDict()
        expand3=max(1,math.floor(prop3*k))
        expand1=k-expand3
        for i in range(num_subunits):
            self.subunit_dict["subfire"+str(i)]=ResFire.from_configure(FireConfig(in_channels=k0+k*i, num_squeeze=4*k, num_expand1=expand1, num_expand3=expand3, skip=False   ) )
        for name, unit in self.subunit_dict.items():
            self.add_module(name,unit)

    def forward(self, x):
        xlist=[]
        for i, (name, unit) in  enumerate(self.subunit_dict.items()):
            xlist.append(x)
            x=unit(torch.cat(xlist,dim=1)) #cat along filter dimension
        return x




SqueezeNetConfig=collections.namedtuple("SqueezeNetConfig","in_channels, base, incr, prop3, freq, sr, out_dim, skipmode,  dropout_rate, num_fires, pool_interval, conv1_stride, conv1_size, pooling_count_offset, num_conv1_filters,  dense_fire_k,  dense_fire_depth_list, dense_fire_compression_level, mode, use_excitation, excitation_r, pool_interval_mode, multiplicative_incr, local_dropout_rate")
class SqueezeNet(serialmodule.SerializableModule):
    '''
        Used ideas from
        -Squeezenet by Iandola et al.
        -Resnet by He et al.
        -densenet by Huang et al.

    '''
    @staticmethod
    def default(in_size,out_dim):
        config=SqueezeNetConfig(in_channels=in_size, base=128, incr= 128, prop3=0.5, freq= 2, sr=0.125, out_dim = out_dim)
        return SqueezeNet(config)
    
    @staticmethod
    def from_args(args):
        if args.fire_skip_mode == "simple":
            skipmode = FireSkipMode.SIMPLE
        elif args.fire_skip_mode == "none":
            skipmode = FireSkipMode.NONE
        if args.squeezenet_dense_fire_depths=="default":
            depthlist=[6, 12, 24, 16]
        elif args.squeezenet_dense_fire_depths=="shallow":
            depthlist=[1, 2, 4, 8]
        else:
            depthlist=None

        config=SqueezeNetConfig(in_channels=args.squeezenet_in_channels,
                base=args.squeezenet_base, incr= args.squeezenet_incr,
                prop3=args.squeezenet_prop3, freq=args.squeezenet_freq,
                sr= args.squeezenet_sr, out_dim=args.squeezenet_out_dim, skipmode=skipmode,
                dropout_rate=args.squeezenet_dropout_rate,
                num_fires=args.squeezenet_num_fires,
                pool_interval=args.squeezenet_pool_interval,
                conv1_stride=args.squeezenet_conv1_stride,
                conv1_size=args.squeezenet_conv1_size,
                pooling_count_offset=args.squeezenet_pooling_count_offset,
                num_conv1_filters=args.squeezenet_num_conv1_filters,
                mode= args.squeezenet_mode,
                dense_fire_k=args.squeezenet_dense_k,
                dense_fire_depth_list= depthlist,
                dense_fire_compression_level=args.squeezenet_dense_fire_compress_level,
                use_excitation=args.squeezenet_use_excitation,
                excitation_r=args.squeezenet_excitation_r,
                pool_interval_mode=args.squeezenet_pool_interval_mode,
                multiplicative_incr=args.squeezenet_multiplicative_incr,
                local_dropout_rate = args.squeezenet_local_dropout_rate
                )
        return SqueezeNet(config)

    def __init__(self, config):
        super().__init__()
        assert(config.skipmode == FireSkipMode.NONE or config.skipmode == FireSkipMode.SIMPLE)
        if config.mode == "densefire":
            logging.info("Making a dense squeezenet.")
            assert config.num_fires == len(config.dense_fire_depth_list)
        num_fires=config.num_fires #8
        first_layer_num_convs=config.num_conv1_filters
        first_layer_conv_width=config.conv1_size
        first_layer_padding= first_layer_conv_width // 2  

        pool_offset=config.pooling_count_offset

        if config.mode != "normal":
            layer_dict=collections.OrderedDict([
            ("conv1", nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride)),
            ]) 
        else:
            layer_dict=collections.OrderedDict([
            ("conv1", nn.Conv2d(config.in_channels, first_layer_num_convs, first_layer_conv_width, padding=first_layer_padding, stride=config.conv1_stride)),
            ("conv1relu", nn.LeakyReLU()),
            ("maxpool1", nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
            ]) 

        self.channel_counts=[ first_layer_num_convs]#initial number of channels entering ith fire layer (labeled i+2 to match paper)
        for i in range(num_fires):
            if  config.mode != "dense_fire":
                if config.pool_interval_mode == "add":
                    e=config.base+config.incr*math.floor(i/config.freq)
                elif config.pool_interval_mode == "multiply":
                    e=config.base* (config.multiplicative_incr ** math.floor(i/config.freq)) 
                num_squeeze=max(math.floor(config.sr*e),1)
                num_expand3=max(math.floor(config.prop3*e),1)
                num_expand1=e-num_expand3
                if config.skipmode == FireSkipMode.SIMPLE and e == self.channel_counts[i]:
                    skip_here=True
                    logging.info("Making simple skip layer.")
                else:
                    skip_here=False
                if config.mode == "wide_resfire":
                    name="wide_resfire{}".format(i+2)
                    to_addi=WideResFire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here, local_dropout_rate=config.local_dropout_rate ))
                elif config.mode == "resfire":
                    name="resfire{}".format(i+2)
                    to_add=ResFire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here ))
                else:
                    name="fire{}".format(i+2)
                    to_add=Fire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here ))

                if config.use_excitation:
                    to_add.skip=False 
                    to_add=ExcitationFire(to_add, in_channels=self.channel_counts[i], out_channels=e, r=config.excitation_r, skip=skip_here)
                    name="ExcitationFire{}".format(i+2)
                layer_dict[name]=to_add

                self.channel_counts.append(e)

            elif config.mode == "dense_fire": 
                    layer_dict["dense_fire{}".format(i+2)]=DenseFire(k0=self.channel_counts[i], num_subunits=config.dense_fire_depth_list[i], k=config.dense_fire_k, prop3=config.prop3 )
                    ts=max(math.floor(config.dense_fire_compression_level*config.dense_fire_k),1)
                    layer_dict["transition{}".format(i+2)]=nn.Conv2d(config.dense_fire_k, ts, 1)
                    self.channel_counts.append(ts)


            if (i+pool_offset) % config.pool_interval == 0:
                layer_dict["maxpool{}".format(i+2)]= nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        layer_dict["dropout"]=nn.Dropout(p=config.dropout_rate)
        if config.mode != "normal":
            layer_dict["final_convrelu"]=nn.LeakyReLU()
            layer_dict["final_conv"]=nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1) 
        else: 
            layer_dict["final_conv"]=nn.Conv2d(self.channel_counts[-1], config.out_dim, kernel_size=1) 
            layer_dict["final_convrelu"]=nn.LeakyReLU()
        self.sequential=nn.Sequential(layer_dict)

    def forward(self, x):
        '''
            Args:
                -x should be batchsize by channels by height by wifth
            reutrns:
                -oput is batchsize by config.outdim
        '''
        out=self.sequential(x)
        out=torch.mean(out,dim=3)
        out=torch.mean(out,dim=2)
        return out




