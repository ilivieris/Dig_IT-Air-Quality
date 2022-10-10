# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] =0., device = 'cpu'):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))
        self.device  = device
        
    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.cat([X, ones], dim=-1)

        if n_samples >= n_dim:
            X = X.to( self.device )
            Y = Y.to( self.device )
            
            # standard
            A = torch.bmm(torch.transpose(X, 1, 2), X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(torch.transpose(X, 1, 2), Y)
            
            weights = torch.Tensor( np.linalg.solve( A.cpu().detach().numpy(), B.cpu().detach().numpy()) ).to( self.device )
        else:
            # Woodbury
            A = torch.bmm(X, torch.transpose(X, 1, 2))
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            temp = torch.Tensor( np.linalg.solve( A.cpu().detach().numpy(), Y.cpu().detach().numpy())).to( self.device )
            weights = torch.bmm(torch.transpose(X, 1, 2), temp )

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)
