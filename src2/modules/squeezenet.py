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
    parser.add_argument("--squeezenet_prop3",type=float, default=0.5)
    parser.add_argument("--squeezenet_freq",type=int, default=2)
    parser.add_argument("--squeezenet_sr",type=float, default=0.125)
    parser.add_argument("--squeezenet_out_dim",type=int)
    parser.add_argument("--squeezenet_resfire",action="store_true")
    parser.add_argument("--squeezenet_wide_resfire",action="store_true")
    parser.add_argument("--squeezenet_dropout_rate",type=float)
    parser.add_argument("--fire_skip_mode", type=str, choices=["simple", "none"], default= "none")
    parser.add_argument("--squeezenet_pool_interval",type=int, default=4)
    parser.add_argument("--squeezenet_num_fires", type=int, default=8)
    parser.add_argument("--squeezenet_conv1_stride", type=int, default=2)
    parser.add_argument("--squeezenet_conv1_size",type=int, default=7)
    parser.add_argument("--squeezenet_num_conv1_filters", type=int, default=96) #should be great than 0, otherwise you get a pool in the first layer
    parser.add_argument("--squeezenet_pooling_count_offset", type=int, default=2) #should be great than 0, otherwise you get a pool in the first layer

    


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
        return WideResFire(in_channels= configure.in_channels, num_squeeze= configure.num_squeeze,num_expand1= configure.num_expand1,num_expand3= configure.num_expand3, skip=configure.skip)


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

class FireSkipMode(Enum):
    NONE=0
    SIMPLE=1


SqueezeNetConfig=collections.namedtuple("SqueezeNetConfig","in_channels, base, incr, prop3, freq, sr, out_dim, skipmode, resfire, dropout_rate, num_fires, pool_interval, conv1_stride, conv1_size, pooling_count_offset, wide_resfire, num_conv1_filters")
class SqueezeNet(serialmodule.SerializableModule):
    '''
        Based on Squeezenet by Iandola et al.
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
        config=SqueezeNetConfig(in_channels=args.squeezenet_in_channels, base=args.squeezenet_base, incr= args.squeezenet_incr, prop3=args.squeezenet_prop3, freq=args.squeezenet_freq, sr= args.squeezenet_sr, out_dim=args.squeezenet_out_dim, skipmode=skipmode, resfire=args.squeezenet_resfire, dropout_rate=args.squeezenet_dropout_rate, num_fires=args.squeezenet_num_fires, pool_interval=args.squeezenet_pool_interval, conv1_stride=args.squeezenet_conv1_stride, conv1_size=args.squeezenet_conv1_size, pooling_count_offset=args.squeezenet_pooling_count_offset, wide_resfire=args.squeezenet_wide_resfire, num_conv1_filters=args.squeezenet_num_conv1_filters  )
        return SqueezeNet(config)

    def __init__(self, config):
        super().__init__()
        assert(config.skipmode == FireSkipMode.NONE or config.skipmode == FireSkipMode.SIMPLE)
        num_fires=config.num_fires #8
        first_layer_num_convs=config.num_conv1_filters
        first_layer_conv_width=config.conv1_size
        first_layer_padding= first_layer_conv_width // 2  

        pool_offset=config.pooling_count_offset

        if config.resfire or config.wide_resfire:
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
            e=config.base+config.incr*math.floor(i/config.freq)
            num_squeeze=max(math.floor(config.sr*e),1)
            num_expand3=max(math.floor(config.prop3*e),1)
            num_expand1=e-num_expand3
            if config.skipmode == FireSkipMode.SIMPLE and e == self.channel_counts[i]:
                    skip_here=True
                    logging.info("Making simple skip layer.")
            else:
                    skip_here=False
            if config.wide_resfire:
                layer_dict["wide_resfire{}".format(i+2)]=WideResFire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here ))
            elif config.resfire:
                layer_dict["resfire{}".format(i+2)]=ResFire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here ))
                
            else:
             
                layer_dict["fire{}".format(i+2)]=Fire.from_configure(FireConfig(in_channels=self.channel_counts[i], num_squeeze=num_squeeze, num_expand1=num_expand1, num_expand3=num_expand3, skip=skip_here ))


            self.channel_counts.append(e)

            if (i+pool_offset) % config.pool_interval == 0:
                layer_dict["maxpool{}".format(i+2)]= nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        layer_dict["dropout"]=nn.Dropout(p=config.dropout_rate)
        if config.resfire:
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




