# Author: Karl Stratos (me@karlstratos.com)
import math
import torch
import torch.nn as nn

from torch.distributions.multivariate_normal import MultivariateNormal


class DoE(nn.Module):

    def __init__(self, dim, hidden, layers, pdf):
        super(DoE, self).__init__()
        self.qY = PDF(dim, pdf)
        self.qY_X = ConditionalPDF(dim, hidden, layers, pdf)

    def forward(self, X, Y, XY_package):
        hY = self.qY(Y)
        hY_X = self.qY_X(Y, X)

        loss = hY + hY_X
        mi_loss = hY_X - hY
        return (mi_loss - loss).detach() + loss


class ConditionalPDF(nn.Module):

    def __init__(self, dim, hidden, layers, pdf):
        super(ConditionalPDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.X2Y = FF(dim, hidden, 2 * dim, layers)

    def forward(self, Y, X):
        mu, ln_var = torch.split(self.X2Y(X), self.dim, dim=1)
        cross_entropy = compute_negative_ln_prob(Y, mu, ln_var, self.pdf)
        return cross_entropy


class PDF(nn.Module):

    def __init__(self, dim, pdf):
        super(PDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.mu = nn.Embedding(1, self.dim)
        self.ln_var = nn.Embedding(1, self.dim)  # ln(s) in logistic

    def forward(self, Y):
        cross_entropy = compute_negative_ln_prob(Y, self.mu.weight,
                                                 self.ln_var.weight, self.pdf)
        return cross_entropy


class NWJJS(nn.Module):

    def __init__(self, dim, hidden, layers):
        super(NWJJS, self).__init__()
        self.fXY = FF(2 * dim, hidden, 1, layers)
        self.lsig = nn.LogSigmoid()
        self.sp = nn.Softplus()

    def forward(self, X, Y, XY_package):
        N = int(math.sqrt(XY_package.size(0)))
        infs = torch.tensor([float('inf')] * N).to(X.device)

        S = self.fXY(XY_package).view(N, N)

        accept = self.lsig(S).diag().mean()
        reject = self.lsig(-S)
        reject = (reject - reject.diag().diag()).sum() / N / (N - 1)
        js = -(accept + reject)

        nwj = (S - infs.diag()).exp().sum() / N / (N - 1) / math.e - \
              S.diag().mean()

        return (nwj - js).detach() + js


class Interpolated(nn.Module):

    def __init__(self, dim, hidden, layers, aY_type, alpha):
        super(Interpolated, self).__init__()
        assert alpha >= 0 and alpha <= 1
        self.fXY = FF(2 * dim, hidden, 1, layers)
        self.ln_aY = get_ln_score_function(dim, hidden, layers, aY_type)
        self.ln_alpha = math.log(alpha) if alpha > 0 else float('-inf')
        self.ln_one_minus_alpha = math.log(1 - alpha) if alpha < 1 \
                                  else float('-inf')

    def forward(self, X, Y, XY_package):
        N = int(math.sqrt(XY_package.size(0)))

        S = self.fXY(XY_package).view(N, N)
        ln_aY = self.ln_aY(Y).to(X.device)

        joint = self.get_joint_term(S, ln_aY, N)
        loo = self.get_loo_term(S, ln_aY, N)

        return loo - joint - 1

    def get_loo_term(self, S, ln_aY, N):
        infs = torch.tensor([float('inf')] * N).to(S.device)
        ln_sumexp_Y_loo = (torch.logsumexp(S - infs.diag(), 0)
                           - math.log(N - 1)).view(N, 1)
        ln_interpol_loo = torch.cat([self.ln_alpha + ln_sumexp_Y_loo,
                                     self.ln_one_minus_alpha + ln_aY],
                                    dim=1)
        ln_interpol_loo = torch.logsumexp(ln_interpol_loo, 1)

        exp_loo = torch.logsumexp(S - ln_interpol_loo - infs.diag(), 0)
        exp_loo = (exp_loo.exp() / (N - 1)).mean()
        return exp_loo

    def get_joint_term(self, S, ln_aY, N):
        ln_sumexp_Y = (torch.logsumexp(S, 0) - math.log(N)).view(N, 1)
        ln_interpol = torch.cat([self.ln_alpha + ln_sumexp_Y,
                                 self.ln_one_minus_alpha + ln_aY],
                                dim=1)
        ln_interpol = torch.logsumexp(ln_interpol, 1)
        joint = (S - ln_interpol).diag().mean()
        return joint


class CPC(nn.Module):

    def __init__(self, dim, hidden, layers):
        super(CPC, self).__init__()
        self.fXY = FF(2 * dim, hidden, 1, layers)
        self.ce = nn.CrossEntropyLoss()
        self.transpose = False

    def forward(self, X, Y, XY_package):
        N = int(math.sqrt(XY_package.size(0)))
        infs = torch.tensor([float('inf')] * N).to(X.device)

        S = self.fXY(XY_package).view(N, N)
        if self.transpose:
            S = S.t()
        loss = self.ce(S, torch.tensor([i for i in range(N)]).to(X.device))

        return loss - math.log(N)


class SingleSampleEstimator(nn.Module):

    def __init__(self, dim, hidden, layers, estimator_type):
        super(SingleSampleEstimator, self).__init__()
        self.estimator_type = estimator_type
        self.fXY = FF(2 * dim, hidden, 1, layers)

    def forward(self, X, Y, XY_package):
        N = int(math.sqrt(XY_package.size(0)))
        infs = torch.tensor([float('inf')] * N).to(X.device)

        S = self.fXY(XY_package).view(N, N)
        joint = S.diag().mean()
        exp_marginal = (S - infs.diag()).exp().sum() / N / (N - 1)
        return self.squash(exp_marginal) - joint

    def squash(self, exp_marginal):
        if self.estimator_type == 'dv':
            return exp_marginal.log()
        elif self.estimator_type == 'nwj':
            return exp_marginal / math.e
        else:
            raise ValueError('Unknown estimator: %s' % (self.estimator_type))


class MINE(nn.Module):

    def __init__(self, dim, hidden, layers, carry_rate=0.99):
        super(MINE, self).__init__()
        self.fXY = FF(2 * dim, hidden, 1, layers)
        self.carry_rate = carry_rate
        self.ema = None

    def forward(self, X, Y, XY_package):
        N = int(math.sqrt(XY_package.size(0)))
        infs = torch.tensor([float('inf')] * N).to(X.device)

        S = self.fXY(XY_package).view(N, N)
        joint = S.diag().mean()
        exp_marginal = (S - infs.diag()).exp().sum() / N / (N - 1)

        self.ema = exp_marginal.detach() if self.ema is None else \
                   self.carry_rate * self.ema + \
                   (1 - self.carry_rate) * exp_marginal.detach()

        mine_loss = (1 / self.ema) * exp_marginal - joint
        dv_loss = self.ema.log() - joint

        return (dv_loss - mine_loss).detach() + mine_loss


class TUBA(nn.Module):

    def __init__(self, dim, hidden, layers, aX_type):
        super(TUBA, self).__init__()
        self.fXY = FF(2 * dim, hidden, 1, layers)
        self.ln_aX = get_ln_score_function(dim, hidden, layers, aX_type)

    def forward(self, X, Y, XY_package):
        N = int(math.sqrt(XY_package.size(0)))
        infs = torch.tensor([float('inf')] * N).to(X.device)

        S = self.fXY(XY_package).view(N, N) - self.ln_aX(X).to(X.device)
        joint = S.diag().mean()
        exp_marginal = (S - infs.diag()).exp().sum() / N / (N - 1)
        return exp_marginal - joint - 1


class LnConstant(nn.Module):

    def __init__(self, c):
        super(LnConstant, self).__init__()
        self.c = c

    def forward(self, X):
        return torch.tensor([math.log(self.c)] * X.size(0)).view(X.size(0), 1)


class LnStandardNormal(nn.Module):

    def __init__(self, dim):
        super(LnStandardNormal, self).__init__()
        self.pdf = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    def forward(self, X):
        return self.pdf.log_prob(X).view(X.size(0), 1)


class FF(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='tanh', dropout_rate=0, layer_norm=False,
                 residual_connection=False):
        super(FF, self).__init__()
        assert (not residual_connection) or (dim_hidden == dim_input)
        self.residual_connection = residual_connection

        self.stack = nn.ModuleList()
        for l in range(num_layers):
            layer = []

            if layer_norm:
                layer.append(nn.LayerNorm(dim_input if l == 0 else dim_hidden))

            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden,
                                   dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])
            layer.append(nn.Dropout(dropout_rate))

            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden,
                             dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = x + layer(x) if self.residual_connection else layer(x)
        return self.out(x)


class CorrelatedStandardNormals(object):

    def __init__(self, dim, rho, device):
        assert abs(rho) <= 1
        self.dim = dim
        self.rho = rho
        self.pdf = MultivariateNormal(torch.zeros(dim).to(device),
                                      torch.eye(dim).to(device))

    def I(self):
        num_nats = - self.dim / 2 * math.log(1 - math.pow(self.rho, 2)) \
                   if abs(self.rho) != 1.0 else float('inf')
        return num_nats

    def hY(self):
        return 0.5 * self.dim * math.log(2 * math.pi)

    def draw_samples(self, num_samples):
        X, ep = torch.split(self.pdf.sample((2 * num_samples,)), num_samples)
        Y = self.rho * X + math.sqrt(1 - math.pow(self.rho, 2)) * ep
        return X, Y


def compute_negative_ln_prob(Y, mu, ln_var, pdf):
    var = ln_var.exp()

    if pdf == 'gauss':
        negative_ln_prob = 0.5 * ((Y - mu) ** 2 / var).sum(1).mean() + \
                           0.5 * Y.size(1) * math.log(2 * math.pi) + \
                           0.5 * ln_var.sum(1).mean()

    elif pdf == 'logistic':
        whitened = (Y - mu) / var
        adjust = torch.logsumexp(
            torch.stack([torch.zeros(Y.size()).to(Y.device), -whitened]), 0)
        negative_ln_prob = whitened.sum(1).mean() + \
                           2 * adjust.sum(1).mean() + \
                           ln_var.sum(1).mean()

    else:
        raise ValueError('Unknown PDF: %s' % (pdf))

    return negative_ln_prob


def get_ln_score_function(dim, hidden, layers, function_type):
        if function_type == 'e':
            return LnConstant(math.e)
        elif function_type == 'ff':
            return FF(dim, hidden, 1, layers)
        elif function_type == 'lp':
            return LnStandardNormal(dim)
        else:
            raise ValueError('Unknown function type: %s' % (function_type))
