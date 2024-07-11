import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os
from torch.nn.parameter import Parameter
import math
import numpy as np
import cv2


class KPAFlowDec(nn.Module):
    def __init__(self, args, chnn=128):
        super().__init__()
        self.args = args
        cor_planes = 4 * (2 * args.corr_radius + 1) ** 2
        self.C_cor = nn.Sequential(
            nn.Conv2d(cor_planes, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, 3, padding=1),
            nn.ReLU(inplace=True))
        self.C_flo = nn.Sequential(
            nn.Conv2d(2, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        self.C_mo = nn.Sequential(
            nn.Conv2d(192+64, 128-2, 3, padding=1),
            nn.ReLU(inplace=True))

        self.kpa = KPA(args, chnn)
        self.gru = SepConvGRU(hidden_dim=chnn, input_dim=chnn+chnn+chnn)
        self.C_flow = nn.Sequential(
            nn.Conv2d(chnn, chnn*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn*2, 2, 3, padding=1))
        self.C_mask = nn.Sequential(
            nn.Conv2d(chnn, chnn*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn*2, 64*9, 1, padding=0))

    def _mo_enc(self, flow, corr, itr):
        feat_cor = self.C_cor(corr)
        feat_flo = self.C_flo(flow)
        feat_cat = torch.cat([feat_cor, feat_flo], dim=1)
        feat_mo = self.C_mo(feat_cat)
        feat_mo = torch.cat([feat_mo, flow], dim=1)
        return feat_mo

    def forward(self, net, inp, corr, flow, itr, upsample=True):
        feat_mo = self._mo_enc(flow, corr, itr)
        feat_moa = self.kpa(inp, feat_mo, itr)
        inp = torch.cat([inp, feat_mo, feat_moa], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.C_flow(net)

        # scale mask to balence gradients
        mask = .25 * self.C_mask(net)
        return net, mask, delta_flow


class KPA(nn.Module):
    def __init__(self, args, chnn):
        super().__init__()
        self.unfold_type = 'x311'
        if 'kitti' in args.dataset:
            self.sc = 15
        else:
            self.sc = 19

        self.unfold = nn.Unfold(kernel_size=3*self.sc, dilation=1, padding=self.sc, stride=self.sc)
        self.scale = chnn ** -0.5
        self.to_qk = nn.Conv2d(chnn, chnn * 2, 1, bias=False)
        self.to_v = nn.Conv2d(chnn, chnn, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        h_k = (3 * self.sc - 1) / 2
        self.w_prelu = nn.Parameter(torch.zeros(1) + 1/h_k)
        self.scp = 0.02
        self.b = 1.

    def _FS(self, attn, shape):
        b, c, h, w, h_sc, w_sc = shape
        device = attn.device
        k = int(math.sqrt(attn.shape[1]))
        crd_k = torch.linspace(0, k-1, k).to(device)
        x = crd_k.view(1, 1, k, 1, 1).expand(b, 1, k, h, w)
        y = crd_k.view(1, k, 1, 1, 1).expand(b, k, 1, h, w)

        sc = torch.tensor(self.sc).to(device)
        idx_x = sc.view(1, 1, 1, 1, 1).expand(b, 1, 1, h, w)
        idx_y = sc.view(1, 1, 1, 1, 1).expand(b, 1, 1, h, w)
        crd_w = torch.linspace(0, w-1, w).to(device)
        crd_h = torch.linspace(0, h-1, h).to(device)
        idx_x = idx_x + crd_w.view(1, 1, 1, 1, w).expand(b, 1, 1, h, w) % self.sc
        idx_y = idx_y + crd_h.view(1, 1, 1, h, 1).expand(b, 1, 1, h, w) % self.sc

        half_ker = torch.tensor(self.sc * 2).to(device)
        o_x = -1 * F.prelu(abs(x - idx_x) - half_ker, self.w_prelu * self.scp) + self.b
        o_x[o_x < 0] = 0
        o_y = -1 * F.prelu(abs(y - idx_y) - half_ker, self.w_prelu * self.scp) + self.b
        o_y[o_y < 0] = 0
        ker_S = o_x * o_y
        ker_S = ker_S.view(b, k**2, h_sc, self.sc, w_sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, k**2, h_sc*w_sc, self.sc**2)
        return ker_S

    def forward(self, *inputs):
        feat_ci, feat_mi, itr = inputs
        b, c, h_in, w_in = feat_mi.shape

        x_pad = self.sc - w_in % self.sc
        y_pad = self.sc - h_in % self.sc
        feat_c = F.pad(feat_ci, (0, x_pad, 0, y_pad))
        feat_m = F.pad(feat_mi, (0, x_pad, 0, y_pad))
        b, c, h, w = feat_c.shape
        h_sc = h // self.sc
        w_sc = w // self.sc 

        fm = torch.ones(1, 1, h_in, w_in).to(feat_m.device)
        fm = F.pad(fm, (0, x_pad, 0, y_pad))
        fm_k = self.unfold(fm).view(1, 1, -1, h_sc*w_sc)
        fm_q = fm.view(1, 1, h_sc, self.sc, w_sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(1, 1, h_sc*w_sc, self.sc**2)
        am = torch.einsum('b c k n, b c n s -> b k n s', fm_k, fm_q)
        am = (am - 1) * 99.
        am = am.repeat(b, 1, 1, 1)

        if itr == 0:
            feat_q, feat_k = self.to_qk(feat_c).chunk(2, dim=1)
            feat_k = self.unfold(feat_k).view(b, c, -1, h_sc*w_sc)
            feat_k = self.scale * feat_k
            feat_q = feat_q.view(b, c, h_sc, self.sc, w_sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h_sc*w_sc, self.sc**2)
            attn = torch.einsum('b c k n, b c n s -> b k n s', feat_k, feat_q)
            attn = attn + am

            ker_S = self._FS(attn, [b, c, h, w, h_sc, w_sc])
            attn_kpa = ker_S.view(attn.shape) * attn
            self.attn = F.softmax(attn_kpa, dim=1)

        feat_v = self.to_v(feat_m)
        feat_v = self.unfold(feat_v).view(b, c, -1, h_sc*w_sc)
        feat_r = torch.einsum('b k n s, b c k n -> b c n s', self.attn, feat_v)
        feat_r = feat_r.view(b, c, h_sc, w_sc, self.sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h, w)
        feat_r = feat_r[:,:,:h_in,:w_in]

        feat_o = feat_mi + feat_r * self.gamma
        return feat_o


class KPAEnc(nn.Module):
    def __init__(self, args, chnn, sc):
        super().__init__()
        self.sc = sc
        self.unfold = nn.Unfold(kernel_size=3*self.sc, dilation=1, padding=self.sc, stride=self.sc)
        self.scale = chnn ** -0.5
        self.to_qk = nn.Conv2d(chnn, chnn * 2, 1, bias=False)
        self.to_v = nn.Conv2d(chnn, chnn, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.mask_k = True

    def forward(self, inputs):
        feat_i = inputs
        b, c, h_in, w_in = feat_i.shape
        x_pad = self.sc - w_in % self.sc
        y_pad = self.sc - h_in % self.sc
        feat = F.pad(feat_i, (0, x_pad, 0, y_pad)) 
        b, c, h, w = feat.shape
        h_sc = h // self.sc
        w_sc = w // self.sc 

        fm = torch.ones(1, 1, h_in, w_in).to(feat.device)
        fm = F.pad(fm, (0, x_pad, 0, y_pad))
        fm_k = self.unfold(fm).view(1, 1, -1, h_sc*w_sc)
        fm_q = fm.view(1, 1, h_sc, self.sc, w_sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(1, 1, h_sc*w_sc, self.sc**2)
        am = torch.einsum('b c k n, b c n s -> b k n s', fm_k, fm_q)
        am = (am - 1) * 99.
        am = am.repeat(b, 1, 1, 1)

        feat_q, feat_k = self.to_qk(feat).chunk(2, dim=1)
        feat_k = self.unfold(feat_k).view(b, c, -1, h_sc*w_sc)
        feat_k = self.scale * feat_k
        feat_q = feat_q.view(b, c, h_sc, self.sc, w_sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h_sc*w_sc, self.sc**2)
        attn = torch.einsum('b c k n, b c n s -> b k n s', feat_k, feat_q)

        attn = attn + am
        self.attn = F.softmax(attn, dim=1)

        feat_v = self.to_v(feat)
        feat_v = self.unfold(feat_v).view(b, c, -1, h_sc*w_sc)
        feat_r = torch.einsum('b k n s, b c k n -> b c n s', self.attn, feat_v)
        feat_r = feat_r.view(b, c, h_sc, w_sc, self.sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h, w)
        feat_r = feat_r[:,:,:h_in,:w_in]
        feat_o = feat_i + feat_r * self.gamma
        return feat_o


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
