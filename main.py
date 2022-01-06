from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import sys
sys.path.append('../Datasets')

import time
import math
import numpy as np
import ALA_nonmon_mon as ott
# import ALA_mon as ott
# import ALA_nonmon as ott
import torch
import matplotlib.pyplot as plt
import importlib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

torch.set_default_dtype(torch.double)
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch.nn as nn
import torch.nn.functional as F

class dataset:
    data = []
    Xtr = []
    ytr = []

    def __init__(self,name,random_state):
        self.name = name
        self.random_state = random_state
        self.data = importlib.import_module(name)
        self.Xtr, Xts, self.ytr, yts = train_test_split(self.data.X_train,
                                        self.data.y_train,train_size=0.80,
                                        random_state=random_state)
        # self.Xtr, self.ytr = shuffle(self.data.X_train, self.data.y_train, random_state=random_state)

trunc = ["Nash", "Dembo"]

curv = ["Curv", "Nocurv"]

list_data_names = ["adult", "ailerons", "appliances", "arcene", "blogfeed",
                    "boston_house", "breast_cancer", "cifar10", "gisette",
                    "iris", "mnist", "mv", "qsar"]

random_seeds = [1, 2, 3, 4]

neurons = [30, 40, 50]

for t in trunc:
    for c in curv:
        fid = open("Results/" + t + "_" + c + '.txt','a')
        fid.truncate(0)
        print('         Name &  perm &  nneu &     K &  KTOT &    Nf &     Ng &      Nint &            f &       g_norm &      time',file=fid)
        fid.close()
        for name in list_data_names:
            for r in random_seeds:
                for nneu in neurons:
                    
                    data = dataset(name,r)
                    # print("Test ", name, " Seed ", r, "Neurons ", nneu, " ", c, " ", t)
                    
                    X_train, y_train = map(torch.tensor, (data.Xtr, data.ytr))

                    X_train = X_train.double()
                    y_train = y_train.double()
                    # y_train = y_train.view(-1, 1).long()
                    
                    X_train = X_train.to(device)
                    y_train = y_train.to(device)
                    
                    ntrain,input_dim = X_train.shape
                    ntrain,out_dim   = y_train.shape
                    
                    # print(X_train.shape)
                    # print(y_train.shape)
                    # print(out_dim)
                    
                    torch.cuda.manual_seed(r)
                    torch.manual_seed(r)
                    
                    
                    class Net(nn.Module):
                    
                        def __init__(self, dims):
                            super(Net, self).__init__()
                    
                            self.nhid = len(dims)
                            self.dims = dims
                    
                            self.fc = nn.ModuleList().double()
                            for i in range(self.nhid - 1):
                                linlay = nn.Linear(dims[i], dims[i+1]).double().to(device)
                                # linlay = nn.Linear(dims[i], dims[i+1], bias = False).double().to(device)
                                self.fc.append(linlay)
                    
                        def forward(self, x):
                            b = torch.tensor([], device = device, dtype=torch.double)
                    
                            for i in range(self.nhid - 2):
                                x = self.fc[i](x)
                                #x = torch.relu(x)
                                # x = torch.sigmoid(x)
                                x = torch.tanh(x)
                                # x = swish(x)
                    
                            x = self.fc[self.nhid - 2](x)
                            # x = torch.softmax(x, dim = 1)
                            b = torch.cat((b,x))
                            return b
                       
                    def swish(x):
                        return torch.mul(x, sigmoid(x))
                    
                    def sigmoid(x):
                        return torch.where(x >= 0, _positive_sigm(x), _negative_sigm(x))
                    
                    def _negative_sigm(x):
                        expon = torch.exp(-x)
                        return 1 / (1 + expon)
                    
                    def _positive_sigm(x):
                        expon = torch.exp(x)
                        return expon / (1 + expon)
                    
                    def init_weights(m):
                        if type(m) == nn.Linear:
                            torch.nn.init.uniform_(m.weight.data,a=-1.0,b=1.0).double()
                            torch.nn.init.uniform_(m.bias.data,a=-1.0,b=1.0).double()
                    
                    def cross_entropy(y_hat, y):
                        return torch.mean(- torch.log(y_hat[range(len(y_hat)), y.view(-1,)])).double()
                    
                    MSELoss = torch.nn.MSELoss()
                    
                    def my_loss(X, y):
                        y_hat = net(X).double()
                        loss = MSELoss(y_hat,y)
                        # loss = cross_entropy(y_hat, y)
                        return loss
                    
                    def my_loss_reg(X, y, ro):
                        y_hat = net(X).double()
                        loss = MSELoss(y_hat,y)
                        # loss = cross_entropy(y_hat, y)
                        l2_reg = torch.tensor(0.0, device = device, dtype=torch.double)
                        for param in net.parameters():
                             l2_reg += torch.norm(param)**2
                        loss += ro * l2_reg
                        return loss
                    
                    
                    #################################
                    # define the variable array for
                    # NWTNM optimizer
                    #################################
                    def dim():
                     	n = 0
                     	for k,v in net.state_dict().items():
                             n += v.numel()
                     	return n
                    
                    def startp(n1):
                        x = torch.zeros(n1,dtype=torch.double,requires_grad=True)
                        torch.nn.init.normal_(x).double()
                    
                        return x.detach().to(device)
                    
                    def set_x(x):
                     	state_dict = net.state_dict()
                     	i = 0
                     	for k,v in state_dict.items():
                             lpart = v.numel()
                             x[i:i+lpart] = state_dict[k].reshape(lpart).double()
                             i += lpart
                    
                    def funct(x):
                        state_dict = net.state_dict()
                        i = 0
                        for k,v in state_dict.items():
                            lpart = v.numel()
                            state_dict[k] = x[i:i+lpart].reshape(v.shape).double()
                            i += lpart
                        net.load_state_dict(state_dict)
                        l_train = my_loss(X_train, y_train)
                        # l_train = my_loss_reg(X_train, y_train, l2_lambda)
                        return l_train
                    
                    def grad(x):
                        for param in net.parameters():
                            if param.requires_grad:
                                if not type(param.grad) is type(None):
                                    param.grad.zero_()
                                param.requires_grad_()
                        f = funct(x)
                        f.backward()
                    
                        if False:
                            g = x.clone().detach()
                            i = 0
                            for v in net.parameters():
                                if v.requires_grad:
                                    lpart = v.numel()
                                    d = v.grad.reshape(lpart)
                                    g[i:i+lpart] = d
                                    i += lpart
                    
                        views = []
                        for p in net.parameters():
                            if p.requires_grad:
                                view = p.grad.view(-1)
                            views.append(view)
                    
                        g1 = torch.cat(views, 0).to(device)
                    
                        return g1
                    
                    def hessdir2(x,d):
                     	if False:
                     	    state_dict = net.state_dict()
                     	    i = 0
                     	    for k,v in state_dict.items():
                     	        lpart = v.numel()
                     	        state_dict[k] = x[i:i+lpart].reshape(v.shape).double()
                     	        i += lpart
                     	    net.load_state_dict(state_dict)
                     	for param in net.parameters():
                             if param.requires_grad:
                                 if not type(param.grad) is type(None):
                                     param.grad.zero_()
                                 param.requires_grad_()
                     	grads = torch.autograd.grad(outputs=funct(x), inputs=net.parameters(), create_graph=True)
                     	dot   = nn.utils.parameters_to_vector(grads).mul(d).sum()
                     	grads = [g.contiguous() for g in torch.autograd.grad(dot, net.parameters(), retain_graph = True)]
                     	return nn.utils.parameters_to_vector(grads)
                    
                    '''
                    in hessdir3 a seconda del valore di goth:
                     FALSE -> si calcola gradstore e lo si memorizza
                     TRUE  -> si usa gradstore salvato senza ricalcolarlo
                    '''
                    def hessdir3(x,d,goth):
                    	for param in net.parameters():
                    		if param.requires_grad:
                    			if not type(param.grad) is type(None):
                    				param.grad.zero_()
                    			param.requires_grad_()
                    	if not goth:
                    		hessdir3.gradstore = torch.autograd.grad(outputs=funct(x), inputs=net.parameters(), create_graph=True)
                    	dot   = nn.utils.parameters_to_vector(hessdir3.gradstore).mul(d).sum()
                    	grads = [g.contiguous() for g in torch.autograd.grad(dot, net.parameters(), retain_graph = True)]
                    	return nn.utils.parameters_to_vector(grads)
                    
                    # which_algo    = 'lbfgs'
                    # which_algo    = 'sgd'
                    which_algo    = 'troncato'
                    maxiter       = 10000
                    maxtim        = 1800
                    l2_lambda     = 1e-05
                    nrnd          = 1
                    tolmax        = 1.e-5
                    tolchmax      = 1.e-9
                    iprint        = 0  # -1
                    satura        = True
                    hd_exact      = True
                    
                    # print()
                    # print("----------------------------------------------")
                    # print(" define a neural net to be minimized ")
                    # print("----------------------------------------------")
                    # print()
                    
                    # n_class = 10
                    # dims      = [input_dim, hidden_1, n_class]
                    dims      = [input_dim, nneu, out_dim]
                    net       = Net(dims).double().to(device)
                    
                    for irnd in range(nrnd):
                        net.apply(init_weights)
                    
                        # print(net)
                        # print(net.parameters())
                    
                        n             = dim()
                        x             = startp(n)
                        set_x(x)
                        l_train       = funct(x)
                        nabla_l_train = grad(x)
                        gnorm         = nabla_l_train.norm().item()
                    
                        # print("numero di parametri totali: n=", n, "nneu: ", nneu,
                        #       "loss: ", l_train.item(), " gloss: ", gnorm)
                    
                        tol    = tolmax
                        tolch  = tolchmax
                    
                        with tqdm(total=maxiter) as pbar:
                            ng = 0
                            ni = 0
                            def fun_closure(x):
                                global ni
                                deltai = ott.n_iter - ni
                                pbar.update(deltai)
                                ni = ott.n_iter
                                l_train = funct(x)
                                return l_train
                    
                            def closure():
                                global ng
                                global ni
                                optimizer.zero_grad()
                                loss1 = my_loss(X_train, y_train)
                                # loss1 = my_loss_reg(X_train, y_train, l2_lambda)
                                ng += 1
                                deltai = optimizer.state_dict()['state'][0]['n_iter'] - ni
                                pbar.update(deltai)
                                ni = optimizer.state_dict()['state'][0]['n_iter']
                                loss1.backward()
                                return loss1
                    
                            def closure_sgd(ni):
                                optimizer.zero_grad()
                                loss1 = my_loss(X_train, y_train)
                                # loss1 = my_loss_reg(X_train, y_train, l2_lambda)
                                pbar.update(1)
                                loss1.backward()
                                return loss1
                    
                            if which_algo == 'lbfgs':
                                timelbfgs = time.time()
                                optimizer = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=maxiter, max_eval=None, tolerance_grad=tol,
                                                              tolerance_change=tolch, history_size=10, line_search_fn="strong_wolfe")
                    
                                optimizer.step(closure)
                                niter = optimizer.state_dict()['state'][0]['n_iter']
                                timelbfgs_tot = time.time() - timelbfgs
                                timeparz = timelbfgs_tot
                            elif which_algo == 'troncato':
                                ott.n_iter = 0
                                #fstar,xstar,niter,nf,ng,nneg,timeparz = ott.NWTNM(fun_closure,grad,hessdir3,x,tol,maxiter,maxtim,iprint,satura,hd_exact)
                                fstar,xstar,niter,nf,ng,nneg,timeparz = ott.NWTNM(funct,grad,hessdir3,x,tol,maxiter,maxtim,iprint,satura,hd_exact,name,r,nneu,c,t)
                            elif which_algo == 'sgd':
                                optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
                    
                                timesgd = time.time()
                                niter = maxiter
                                for it in range(0, niter):
                                    closure_sgd(it)
                                    optimizer.step()
                                timeparz = time.time() - timesgd
                    
                        pbar.close()
                    
                        # print("KTOT:",niter,"time:",timeparz,"fstar:",fstar.item())
