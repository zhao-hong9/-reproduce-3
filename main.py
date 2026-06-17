import os
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch import nn
from time import time
from tqdm import tqdm
from model.SSAF import SSAF
from utils.FCLSU import FCLSU
from utils.loadhsi import loadhsi
from utils.hyperVca import hyperVca
from utils.result_em import result_em

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('training on', device)

cases = ['ex2', 'ridge', 'houston']
case = cases[1]
rCol = 100
nCol = 100
epochs = 2000
z_dim = 4


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)

def train(lr=0.005, lambda_y2=0.04, lambda_kl=0.001, lambda_pre=10, lambda_sad=5, lambda_vol=10):

    Y, A_true, P, M_true = loadhsi(case)
    vca_em, _, snrEstimate = hyperVca(Y, P)
    vca_em_l = vca_em.T
    M0 = np.reshape(vca_em_l, [1, vca_em_l.shape[1], vca_em_l.shape[0]]).astype('float32')
    M0 = torch.tensor(M0).to(device)
    print('SNR:', snrEstimate)

    Channel, N = Y.shape
    batch_size = 1

    FCLS_a = FCLSU(vca_em, Y, 0.01)
    FCLS_a = FCLS_a.clone().detach()
    FCLS_a = torch.reshape(FCLS_a, (P, rCol, nCol)).unsqueeze(0)

    Y = torch.reshape(torch.tensor(Y), (Channel, rCol, nCol)).unsqueeze(0)
    train_cube = Y
    train_cube = torch.utils.data.TensorDataset(train_cube, FCLS_a)
    train_cube = torch.utils.data.DataLoader(train_cube, batch_size=batch_size, shuffle=True)

    model = SSAF(P, Channel, rCol, nCol, z_dim, M0).to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tic = time()
    losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        for step, (y, fcls_a) in enumerate(train_cube):
            y = y[0].unsqueeze(0).to(device)
            fcls_a = fcls_a[0].reshape(P, N).to(device)
            first_a, second_a, first_y, second_y, em_tensor, mu_s, mu_d, var_s, var_d = model(y)
            y = y.permute(2, 3, 0, 1)
            y = y.reshape(rCol * nCol, Channel)

            loss_y_1 = ((first_y - y) ** 2).sum() / y.shape[0]
            loss_y_2 = ((y - second_y) ** 2).sum() / y.shape[0]

            loss_rec = loss_y_1 + lambda_y2 * loss_y_2

            loss_kl = -0.5 * (var_s + 1 - mu_s ** 2 - var_s.exp())
            loss_kl = loss_kl.sum() / y.shape[0]
            loss_kl = torch.max(loss_kl, torch.tensor(0.2).to(device))
            loss_a1_a2 = ((first_a - second_a) ** 2).sum() / first_a.shape[0]

            if epoch < epochs // 2:
                loss_a = (first_a.T - fcls_a).square().sum() / y.shape[0]
                loss = loss_rec + lambda_kl * loss_kl + lambda_pre * loss_a + 0.1 * loss_a1_a2

            else:
                em_bar = em_tensor.mean(dim=1, keepdim=True)
                loss_minvol = ((em_tensor - em_bar) ** 2).sum() / y.shape[0] / P / Channel

                em_bar = em_tensor.mean(dim=0, keepdim=True)
                aa = (em_tensor * em_bar).sum(dim=2)
                em_bar_norm = em_bar.square().sum(dim=2).sqrt()
                em_tensor_norm = em_tensor.square().sum(dim=2).sqrt()

                sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
                loss_sad = sad.sum() / y.shape[0] / P

                loss = loss_rec + lambda_kl * loss_kl + lambda_vol * loss_minvol + lambda_sad * loss_sad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().numpy())

    toc = time()

    model.eval()
    with torch.no_grad():
        first_a, second_a, first_y, second_y, EM_hat, mu_s, mu_d, var_s, var_d = model(Y.to(device))

        Y_hat = first_y.cpu().numpy()
        A_hat = first_a.cpu().numpy().T

        Y = Y.permute(2, 3, 0, 1).cpu().numpy()
        Y = Y.reshape(rCol * nCol, Channel)

        dev = np.zeros([P, P])
        for i in range(P):
            for j in range(P):
                dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
        pos = np.argmin(dev, axis=0)

        A_hat = A_hat[pos, :]
        EM_hat = EM_hat[:, pos]
        if case == 'ex2':
            EM_hat = EM_hat.data.cpu().numpy()
            EM_hat = np.transpose(EM_hat, (2, 1, 0))
            Mvs = np.reshape(M_true, [Channel, P * N])
            EM_hat = np.reshape(EM_hat, [Channel, P * N])
            armse_y, asad_y, armse_a, armse_em, asad_em = result_em(EM_hat, Mvs, A_hat, A_true, Y, Y_hat)
        else:
            EM_hat = EM_hat.data.cpu().numpy()
            EM_hat = np.transpose(EM_hat, (2, 1, 0))
            armse_y, asad_y, armse_a, armse_em, asad_em = result_em(EM_hat, M_true, A_hat, A_true, Y, Y_hat)

        return armse_y, asad_y, armse_a, armse_em, asad_em, toc - tic


if __name__ == '__main__':
    armse_y, asad_y, armse_a, armse_em, asad_em, tim = train()

    print('*' * 70)
    print('time elapsed:', tim)
    print('RESULTS:')
    print('aRMSE_Y:', armse_y)
    print('aSAD_Y:', asad_y)
    print('aRMSE_a:', armse_a)
    print('aRMSE_M', armse_em)
    print('aSAD_em', asad_em)
