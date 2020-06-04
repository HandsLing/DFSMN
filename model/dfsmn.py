#!/usr/bin/env python
# coding=utf-8 
import torch 
import torch.nn as nn

class DFSMN(nn.Module):

    def __init__(
                    self,
                    input_dim,
                    hidden_dim,
                    output_dim,
                    left_frames=1,
                    left_dilation=1, 
                    right_frames=1,
                    right_dilation=1, 
        ): 
        ''' 
            input_dim as it's name ....
            hidden_dim means the dimension or channles num of the memory 

            left means history
            right means future

        '''
        super(DFSMN, self).__init__() 
        
        self.in_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False)
        self.left_conv = nn.Sequential(
                        nn.ConstantPad1d([left_dilation*left_frames,-left_dilation],0),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=left_frames,dilation=left_dilation)
                )
        
        self.right_conv = nn.Sequential(
                        nn.ConstantPad1d([-right_dilation,right_frames*right_dilation],0),
                        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=right_frames,dilation=right_dilation)
                )

        self.out_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, inputs, hidden=None):
        out = self.in_conv(inputs)
        left = self.left_conv(out)
        right = self.right_conv(out)
        out = out+left+right 
        if hidden is not None:
            out += hidden 
        out = self.out_conv(out)
        return out 

if __name__ == '__main__':
    inputs = torch.randn(10,257,199) 
    net = DFSMN(257,128,137, left_dilation=2, right_dilation=3) 
    net(inputs)
