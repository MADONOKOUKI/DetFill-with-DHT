import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv( nn.Module ):
   def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1 ):
      super(Conv, self).__init__()
      self.conv    = nn.Conv2d( in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding )
      self.bn      = nn.BatchNorm2d( out_planes )
   def forward(self, x):
      if hasattr(self,"bn"):
         return F.relu( self.bn( self.conv( x ) ), inplace=True )
      else:
         return F.relu( self.conv( x ), inplace=True )
   def bnabsorb(self):
      # Conv
      #cw = self.conv.weight
      #cb = self.conv.bias
      # BN
      e = self.bn.eps
      m = self.bn.running_mean
      v = self.bn.running_var
      #a = self.bn.affine
      w = self.bn.weight.data
      b = self.bn.bias.data

      def tot( vec ):
         cws = self.conv.weight.size()
         tv  = (cws[0], 1, 1, 1)
         tr  = (1, cws[1], cws[2], cws[3])
         return vec.view( *tv ).repeat( *tr )
      self.conv.weight.data *= tot( w / torch.sqrt( v + e ) )
      self.conv.bias.data   -= m
      self.conv.bias.data   *= w / torch.sqrt( v + e )
      self.conv.bias.data   += b

      del self.bn

class SketchNet( nn.Module ):
   def __init__(self):
      super(SketchNet, self).__init__()

      self.padding = nn.ReplicationPad2d(4)
      self.layers = nn.Sequential(
         Conv(  2,  128, 2, 9, 0 ),
         Conv( 128, 128 ),
         Conv( 128, 128 ),
         Conv( 128, 128 ),
         Conv( 128, 128 ),
         Conv( 128, 256, 2 ), # 1/4
         Conv( 256, 256 ),
         Conv( 256, 256 ),
         Conv( 256, 256 ),
         Conv( 256, 512, 2 ), # --> 1/8
         Conv( 512, 512 ),
         Conv( 512, 512 ),
         Conv( 512, 256 ),
         Conv( 256, 256 ),
         Conv( 256, 256 ),
         Conv( 256, 256 ),
         Conv( 256, 256 ),
         Conv( 256, 256 ),
         nn.Dropout2d( 0.2 ),
         nn.PixelShuffle(2),
         Conv(  64, 64 ),
         Conv(  64, 64 ),
         nn.PixelShuffle(2),
         Conv(  16, 16 ),
         Conv(  16, 16 ),
         nn.PixelShuffle(2), 
         Conv(   4,  4 ),
         nn.Conv2d( 4, 1, kernel_size=3, stride=1, padding=1 )
      )
   def forward(self, x):
      return F.sigmoid( self.layers( self.padding( x ) ) )
   def bnabsorb(self):
      for l in self.layers:
         if isinstance(l,Conv):
            l.bnabsorb()

"""
import torch
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = SketchNet()
print( "{:,}".format( count_parameters(model) ) )
"""


