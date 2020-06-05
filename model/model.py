import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]) + '/model')
from show import show_params, show_model
import torch.nn.functional as F
from conv_stft import ConvSTFT, ConviSTFT
from loss import si_snr
from dfsmn import  DFSMN

class Net(nn.Module):

    def __init__(self, rnn_layers=2, rnn_units=128, win_len=400, win_inc=100, fft_len=512, win_type='hanning', mode='TCS'):
        super(Net, self).__init__()

        # for fft 
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type 

        input_dim = win_len
        output_dim = win_len
        
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        fix=True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'real', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'real', fix=fix)
        
        in_dim = self.fft_len//2
        out_dim = self.fft_len//2

        self.fsmn1 = DFSMN(in_dim, rnn_units, out_dim, 3,3,1,1)
        self.fsmn2 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn3 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn4 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn5 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn6 = DFSMN(in_dim, rnn_units, out_dim, 7,3,1,1)
        self.fsmn7 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn8 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn9 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn10 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn11 = DFSMN(in_dim, rnn_units, out_dim, 7,3,0,1)
        self.fsmn12 = DFSMN(in_dim, rnn_units, out_dim, 3,3,0,1)

        show_params(self)


    def forward(self, inputs, lens=None):
        
        specs, phase = self.stft(inputs)
        specs = specs[:,1:] 
        
        out, outh = self.fsmn1(specs)
        out = torch.relu(out)
        out, outh = self.fsmn2(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn3(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn4(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn5(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn6(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn7(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn8(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn9(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn10(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn11(out,outh)
        out = torch.relu(out)
        out, outh = self.fsmn12(out,outh)
        mask = torch.sigmoid(out)
        out_spec = specs*mask
        out_spec = F.pad(out_spec,[0,0,1,0])
        out_wav = self.istft(out_spec, phase)
         
        out_wav = torch.squeeze(out_wav, 1)
        return out_spec,  out_wav

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params

    def loss(self, inputs, labels, loss_mode='SI-SNR'):
       
        if loss_mode == 'MSE':
            b,t,d = inputs.shape 
            #gth_spec = self.stft(labels)
            return F.mse_loss(inputs, labels, reduction='mean')*d

        elif loss_mode == 'SI-SNR':
            #return -torch.mean(si_snr(inputs, labels))
            return -(si_snr(inputs, labels))
        elif loss_mode == 'MAE':
            gth_spec, gth_phase = self.stft(labels) 
            b,d,t = inputs.shape 
            return torch.mean(torch.abs(inputs-gth_spec))*d


if __name__ == '__main__':
    torch.manual_seed(10)
    inputs = torch.randn([10,16000*4]).clamp_(-1,1)
    labels = torch.randn([10,16000*4]).clamp_(-1,1)
    inputs = inputs #labels
    net = Net()
    outputs = net(inputs)[1]
    loss = net.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)

