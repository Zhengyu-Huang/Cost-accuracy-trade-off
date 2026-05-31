import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer

    

def _get_act(act):
    """
    Get activation function by name.
    
    Args:
        act: String name of activation function. Options:
             'tanh', 'gelu', 'relu', 'elu', 'leaky_relu', 'none'
    
    Returns:
        Activation function callable or None
    """
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func


def compute_Fourier_modes_helper(ndims, nks, Ls):
    '''
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k, cannot have 0

        Parameters:  
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[nmodes, ndims]
    '''
    assert(len(nks) == len(Ls) == ndims)    
    if ndims == 1:
        nk, Lx = nks[0], Ls[0]
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nk + 1):
            k_pairs[i, :] = 2*np.pi/Lx*kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2*nx*ny + nx + ny
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if (ky==0 and kx<=0): 
                    continue

                k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4*nx*ny*nz + 2*(nx*ny + nx*nz + ny*nz) + nx + ny + nz
        k_pairs    = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if (kz==0 and (ky<0  or (ky==0 and kx<=0))): 
                        continue

                    k_pairs[i, :] = 2*np.pi/Lx*kx, 2*np.pi/Ly*ky, 2*np.pi/Lz*kz
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")
    
    k_pairs = k_pairs[np.argsort(k_pair_mag, kind='stable'), :]
    return k_pairs


def compute_Fourier_modes(ndims, nks, Ls):
    '''
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

        Parameters:  
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[nmodes, ndims]
    '''
    assert(len(nks) == len(Ls))
    k_pairs = compute_Fourier_modes_helper(ndims, nks, Ls)
    
    return k_pairs


def compute_Fourier_bases(nodes, modes):
    '''
    Compute Fourier bases for the whole space
    Fourier bases are cos(kx), sin(kx), 1

        Parameters:  
            nodes        : float[batch_size, nnodes, ndims]
            modes        : float[nmodes, ndims]
            
        Return :
            bases : float[batch_size, nnodes, nbases]   bases_c, bases_s, bases_0
            
    '''
    # temp : float[batch_size, nnodes, nmodes]
    temp  = torch.einsum("bxd,kd->bxk", nodes, modes) 
    
    bases_c = torch.cos(temp) 
    bases_s = torch.sin(temp) 
    batch_size, nnodes, _ = temp.shape
    bases_0 = torch.ones(batch_size, nnodes, 1, dtype=temp.dtype, device=temp.device)
    bases = torch.cat((bases_c, bases_s, bases_0), dim=-1)
    return bases

################################################################
# Fourier layer
################################################################
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nmodes, _ = modes.shape
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.nmodes, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.nmodes, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, 1, dtype=torch.float
            )
        )


    def forward_separate(self, x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0):
        '''
        Compute Fourier neural layer
            Parameters:  
                x                   : float[batch_size, in_channels, nnodes]
                bases_c, bases_s    : float[batch_size, nnodes, nmodes]
                bases_0             : float[batch_size, nnodes, 1]
                wbases_c, wbases_s  : float[batch_size, nnodes, nmodes]
                wbases_0            : float[batch_size, nnodes, 1]

            Return :
                x                   : float[batch_size, out_channels, nnodes]
        '''    
        x_c_hat =  torch.einsum("bix,bxk->bik", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bxk->bik", x, wbases_s)
        x_0_hat =  torch.einsum("bix,bxk->bik", x, wbases_0)

        weights_c, weights_s, weights_0 = self.weights_c, self.weights_s, self.weights_0
        
        f_c_hat = torch.einsum("bik,iok->bok", x_c_hat, weights_c) - torch.einsum("bik,iokw->bok", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bik,iok->bok", x_s_hat, weights_c) + torch.einsum("bik,iokw->bok", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bik,iok->bok", x_0_hat, weights_0) 

        x = torch.einsum("bok,bxk->box", f_0_hat, bases_0) + 2*torch.einsum("bok,bxk->box", f_c_hat, bases_c) - 2*torch.einsum("bok,bxk->box", f_s_hat, bases_s) 
        
        return x
    
    

    def forward(self, x, bases, wbases):
        '''
        Compute Fourier neural layer
            Parameters:  
                x                   : float[batch_size, in_channels, nnodes]
                bases               : float[batch_size, nnodes, nbases]
                wbases              : float[batch_size, nnodes, nmodes, nbases]

            Return :
                x                   : float[batch_size, out_channels, nnodes]
        ''' 
        batch_size, in_channels, nnodes = x.shape
        out_channels, nmodes = self.out_channels, self.nmodes

        # ------------------------------------------------------------
        # Fused forward DFT
        # ------------------------------------------------------------
        x_hat = torch.bmm(x, wbases)  # (batch_size, in_channels, nbases)

        x_c_hat =  x_hat[:, :, :nmodes] 
        x_s_hat = -x_hat[:, :, nmodes:2 * nmodes] 
        x_0_hat =  x_hat[:, :, 2 * nmodes:] 

        weights_c, weights_s, weights_0 = self.weights_c, self.weights_s, self.weights_0

        # ------------------------------------------------------------
        # Channel mixing
        # ------------------------------------------------------------
        f_c_hat = torch.einsum("bik,iok->bok", x_c_hat, weights_c) - torch.einsum("bik,iok->bok", x_s_hat, weights_s)
        f_s_hat = torch.einsum("bik,iok->bok", x_s_hat, weights_c) + torch.einsum("bik,iok->bok", x_c_hat, weights_s)
        f_0_hat = torch.einsum("bik,iok->bok", x_0_hat, weights_0)

        # ------------------------------------------------------------
        # Fused inverse DFT
        # 
        #  out = f0*b0 + 2*fc*bc - 2*fs*bs
        # ------------------------------------------------------------
        f_hat = torch.cat(
            [
                 2.0 * f_c_hat.reshape(batch_size, out_channels, nmodes),
                -2.0 * f_s_hat.reshape(batch_size, out_channels, nmodes),
                 f_0_hat.reshape(batch_size, out_channels, 1),
            ],
            dim=-1,
        )  # (batch_size, out_channels, nbases)

        out = torch.bmm(f_hat, bases.transpose(1, 2))

        return out



def compute_gradient(f, directed_edges, edge_gradient_weights):
    '''
    Compute gradient of field f at each node
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)   =         :
       :                                :
    xj - x                        f(xj) - f(x)
    
    in matrix form   dx  nable f(x)   = df.
    
    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    directed_edges stores directed edges (x, x1), (x, x2), ..., (x, xj)
    edge_gradient_weights stores its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]

    Then the gradient can be computed 
    gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 
    with scatter_add for each edge
    
    
        Parameters: 
            f : float[batch_size, in_channels, nnodes]
            directed_edges : int[batch_size, max_nedges, 2] 
            edge_gradient_weights : float[batch_size, max_nedges, ndims]
            
        Returns:
            x_gradients : float Tensor[batch_size, in_channels*ndims, max_nnodes]
            * in_channels*ndims dimension is gradient[x_1], gradient[x_2], gradient[x_3]......
    '''

    f = f.permute(0,2,1)
    batch_size, max_nnodes, in_channels = f.shape
    _, max_nedges, ndims = edge_gradient_weights.shape
    # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
    # target\source : int Tensor[batch_size, max_nedges]
    # message : float Tensor[batch_size, max_nedges, in_channels*ndims]

    target, source = directed_edges[...,0], directed_edges[...,1]  # source and target nodes of edges
    message = torch.einsum('bed,bec->becd', edge_gradient_weights, f[torch.arange(batch_size).unsqueeze(1), source] - f[torch.arange(batch_size).unsqueeze(1), target]).reshape(batch_size, max_nedges, in_channels*ndims)
    
    # f_gradients : float Tensor[batch_size, max_nnodes, in_channels*ndims]
    f_gradients = torch.zeros(batch_size, max_nnodes, in_channels*ndims, dtype=message.dtype, device=message.device)
    f_gradients.scatter_add_(dim=1, src=message, index=target.unsqueeze(2).repeat(1,1,in_channels*ndims))
    
    return f_gradients.permute(0,2,1)
    



class PCNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        act="gelu",
    ):
        super(PCNO, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                ndims : int 
                    Dimensionality of the problem
                modes : float[nmodes, ndims]
                    It contains nmodes modes k, and Fourier bases include : cos(k x), sin(k x), 1  
                    * We cannot have both k and -k
                    * k is not integer, and it has the form 2pi*K/L0  (K in Z)
                layers : list of int
                    number of channels of each layer
                    The lifting layer first lifts to layers[0]
                    The first Fourier layer maps between layers[0] and layers[1]
                    ...
                    The nlayers Fourier layer maps between layers[nlayers-1] and layers[nlayers]
                    The number of Fourier layers is len(layers) - 1
                fc_dim : int 
                    hidden layers for the projection layer, when fc_dim > 0, otherwise there is no hidden layer
                in_dim : int 
                    The number of channels for the input function
                    For example, when the coefficient function and locations (a(x, y), x, y) are inputs, in_dim = 3
                out_dim : int 
                    The number of channels for the output function

                act : string (default gelu)
                    The activation function

            
            Returns:
                Point cloud neural operator

        """
        
        self.register_buffer('modes', modes) 
        

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims*in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)
        self.softsign = F.softsign
 

    def forward(self, x, aux):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
            
            Parameters: 
                x : Tensor float[batch_size, max_nnomdes, in_dim] 
                    Input data
                aux : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]  
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]  
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes]  
                                    rho(x)dx used for integrations; padding with 0

                    directed_edges : Tensor int[batch_size, max_nedges, 2]  
                                     direted edge pairs; padding with 0  
                                     gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

                    edge_gradient_weights      : Tensor float[batch_size, max_nedges, ndim] 
                                                 pinvdx on each directed edge 
                                                 gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

            
            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Output data

        """

        length = len(self.ws)

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux
        # bases: float[batch_size, nnodes, nmodes]
        bases  = compute_Fourier_bases(nodes, self.modes) 
        # node_weights: float[batch_size, nnodes]
        # wbases: float[batch_size, nnodes, nmodes]
        # set nodes with zero measure to 0
        wbases = torch.einsum("bxk,bx->bxk", bases, node_weights)
        
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        
        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases, wbases)
            x2 = w(x)
            x3 = gw(self.softsign(compute_gradient(x, directed_edges, edge_gradient_weights)))
            x = x1 + x2 + x3
            if self.act is not None and i != length - 1:
                x = self.act(x) 

        
        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

       
        return x 
    


# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def PCNO_train(x_train, aux_train, y_train, x_test, aux_test, y_test, config, model, save_model_name="./PCNO_model", checkpoint_path=None):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    ndims = model.ndims # n_train, size, n_channel
    print("In PCNO_train, ndims = ", ndims)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    myloss = LpLoss(d=1, p=2, size_average=False)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
                     lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['train']['base_lr'], 
            div_factor=2, final_div_factor=100, pct_start=0.2,
            steps_per_epoch=len(train_loader), epochs=config['train']['epochs'])
    
    current_epoch, epochs = 0, config['train']['epochs']
    
    
    for ep in range(current_epoch, epochs):
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in train_loader:
            x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0


        model.eval()
        with torch.no_grad():
            for x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in test_loader:
                x, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

                batch_size_ = x.shape[0]
                out = model(x, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out = out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)
                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        
        # Average losses over dataset size
        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test

        # Store losses
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    

        t2 = default_timer()


        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2, flush=True)
        if ((ep %100 == 99) or (ep == epochs -1)):
            if save_model_name:    
                torch.save(model.state_dict(), save_model_name + ".pth")
                if normalization_x:
                    torch.save(x_normalizer.state_dict(), save_model_name + "_normalization_x.pth")
                if normalization_y:
                    torch.save(y_normalizer.state_dict(), save_model_name + "_normalization_y.pth")
          
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses









