import torchvision.datasets
from torch.nn.parameter import Parameter
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F

class DA_HGNN(nn.Module):
    def __init__(self,top_k,num_feature,sigma,multi_head):
        super(DA_HGNN, self).__init__()
        self.multi_head = multi_head
        self.attentions = [DA_HGAN(in_features=256,sigma=sigma,multi_head=multi_head) for _ in
                           range(multi_head)]   #多头注意力
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.net_SingleHGCN = SingleHGCN(top_k=top_k, num_feature=num_feature)
        self.net_DA_HGAN_2 = DA_HGAN(sigma=sigma,in_features=256)
    def forward(self,X):
        X,E,H = self.net_SingleHGCN(X)  #单层超图卷积网络
        X = torch.cat([att(X,E,H) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        # print('X',X[:5,:5])
        X_tilde = self.net_DA_HGAN_2(X,E,H)
        # print('X', X_tilde[:5, :5])
        return X_tilde

class SingleHGCN(nn.Module):
    def __init__(self,top_k,num_feature):
        super(SingleHGCN,self).__init__()
        self.theta = Parameter(
            torch.normal(0, 0.01, size=(num_feature,256), requires_grad=True,device=torch.device('cuda:0')))
        self.top_k = top_k+1
    def forward(self,X):
        X = torch.tensor(X, device=torch.device('cuda:0')).reshape(-1, 784)
        H = self.euclideanDistance(X)
        X,E = self.singleHGCN(X,H)
        return (X,E,H)

    # 欧式距离计算出超边关系矩阵
    def euclideanDistance(self,X):
        A = torch.sum(X**2,dim=1).reshape(1,-1)
        B = torch.sum(X**2,dim=1).reshape(-1,1)
        C = torch.mm(X,X.T)
        dist_matric = torch.abs(A+B-2*C)
        values, indices = torch.topk(dist_matric, k=self.top_k, dim=1, largest=False)
        H = torch.zeros_like(dist_matric,device=torch.device('cuda:0'))
        for i, indice in zip(range(dist_matric.shape[0]), indices):
            H[indice, i] = 1.0
        return H
    #单层卷积操作
    def singleHGCN(self,X,H):
        '''
        单层超图卷积网络，用于学习节点特征和超边特征的低纬嵌入
        :param X:初始节点的特征矩阵
        :param H:超图关联矩阵
        :param De:超边度的对角矩阵
        :param Dv:顶点度的对角矩阵

        实验过程中默认节点数n等于超边数m
        :return:
        '''
        Dv = torch.diag(torch.pow(torch.sum(H, dim=1),-1 / 2))
        De = torch.diag(torch.pow(torch.sum(H, dim=0), -1 / 2))
        X = torch.mm(torch.mm(torch.mm(torch.mm(De,H.T),Dv),X),self.theta)   #低维节点特征嵌入X
        E = torch.mm(torch.mm(torch.mm(Dv,H),De),X)           #超边特征嵌入E
        return X,E

class DA_HGAN(nn.Module):
    def __init__(self,sigma,in_features,alpha=0.2,multi_head=1):
        '''
        :param sigma: 相似度的阈值
        :param alpha: LeakyRelu的参数，默认0.2
        '''
        super(DA_HGAN, self).__init__()
        self.W = Parameter(
            torch.zeros(size=(in_features, int(256 / multi_head)),requires_grad=True, device=torch.device('cuda:0')))
        self.alpha_x = Parameter(
            torch.zeros(size=(2 * int(256 / multi_head), 1), requires_grad=True, device=torch.device('cuda:0')))
        self.alpha_e = Parameter(
            torch.zeros(size=(2 * int(256 / multi_head), 1), requires_grad=True, device=torch.device('cuda:0')))
        self.sigma = sigma
        self.net_ELU = nn.ELU()
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self,X,E,H):
        return self.DA_HGAN(X,E,H)

    def DA_HGAN(self,Xi,Ek,H):
        '''
        :param X:低维节点特征嵌入 shape:nxd
        :param E:超边特征嵌入     shape:mxd
        :sigma 是预定义的阈值 超参数
        :return:

        密度感知超图注意网络主要由两部分组成：密度感知注意顶点聚合和密度感知注意超边聚合。
        '''

        rho_xi = self.node_density(Xi,H,self.sigma)  #节点密度
        rho_hyper_edge = self.hyper_edge_density(rho_xi,H)  #超边密度
        E = self.attention(Xi,Ek,rho_xi,node=True,H=H,X=Xi)    #节点的注意力值，node为true表示计算的是节点注意力值
        X = self.attention(Ek,Xi,rho_hyper_edge,node=False,H=H,X=E)   #超边的注意力值

        return X

    #通过余弦相似度计算密度
    def node_density(self,X,H,sigma:float):
        neiji = torch.mm(X, X.T)    #内积
        mochang = torch.sqrt(torch.sum(X * X, dim=1).reshape(1, -1) * (torch.sum(X * X, dim=1).reshape(-1, 1))) #模长
        cosim = neiji/mochang   #余弦相似度矩阵

        #矩阵元素小于sigma的全部置为0，对角线元素也置0，因为不需要自己和自己的相似度
        cosim = torch.where(cosim>sigma,cosim,torch.zeros_like(cosim))\
                *(torch.ones_like(cosim,device='cuda:0')-torch.eye(cosim.shape[0],device='cuda:0'))
        #节点和超边的关系矩阵H的内积的每一行，可以表示每个节点的所有相邻节点
        xx = torch.where(torch.mm(H,H.T) > 0, float(1), float(0)) \
             * (torch.ones_like(cosim,device='cuda:0') - torch.eye(cosim.shape[0],device='cuda:0'))
        #将每个节点与相邻节点的相似度相加，就是该节点的密度
        rho = torch.sum(cosim*xx,dim=1).reshape(-1,1)
        return rho
    #计算节点与边的注意力值,然后计算标准化密度，最后得出注意力权重
    def attention(self,Xi:torch.Tensor,Ek:torch.Tensor,rho:torch.Tensor,node:bool,H,X):
        '''
        :param Xi:节点嵌入矩阵
        :param Ek: 超边嵌入矩阵
        :param rho:节点密度
        :return:注意力权重
        '''
        # 将WX和WE拼接

        a_input = self._prepare_attentional_mechanism_input(
            Xi, Ek)  # 实现论文中的特征拼接操作 Wh_i||Wh_j ，得到一个shape = (N ， N, 2 * out_features)的新特征矩阵
        if node:
            a_x = self.leakyrelu(torch.matmul(a_input, self.alpha_x).squeeze(2))
        else:
            a_x = self.leakyrelu(torch.matmul(a_input,self.alpha_e).squeeze(2))
        #对节点嵌入数据进行线性变换，成为nx1,对超边嵌入数据进行线性变换，成为mx1
        # for Xi_sub in Xi:
        #     for Ek_sub in Ek:
        #         # 上下合并
        #         concat = torch.cat((torch.mm(Xi_sub.reshape(1,-1),self.W).T,torch.mm(Ek_sub.reshape(1,-1),self.W).T),dim=0)
        #         # alpha_x是2dx1的权重矩阵,需要训练？
        #         #判断时边权重还是节点权重
        #         if node:
        #             value = net_leakyRelu(torch.mm(self.alpha_x.T, concat))
        #         else:
        #             value = net_leakyRelu(torch.mm(self.alpha_e.T, concat))
        #         a_x.append(value)    #将所有的xi与ek的注意力值存入a_x
        # # self.alpha_x.retain_grad()
        # a_x = torch.tensor(a_x,device=torch.device('cuda:0')).reshape(Xi.shape[0],-1)     #获得节点x_i和超边e_k之间的注意力值a_x_e
        rho_tilde =torch.tensor([enum/torch.max(rho)*torch.max(a_x) for enum in rho],
                                device=torch.device('cuda:0')).reshape(-1,1)
        a_x_tilde = a_x+rho_tilde
        zero_vec = -1e12 * torch.ones_like(a_x_tilde)  # 将没有连接的边置为负无穷
        attention = torch.where(H > 0, a_x_tilde, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, 0.2, training=self.training)  # dropout，防止过拟合
        if node:
            h_prime = self.net_ELU(torch.matmul(attention, torch.mm(X,self.W)))  # [N, N].[N, out_features] => [N, out_features]
        else:
            h_prime = self.net_ELU(torch.matmul(attention, X))  # [N, N].[N, out_features] => [N, out_features]
        return h_prime

    def hyper_edge_density(self,rho_x:torch.Tensor,H:torch.Tensor):
        '''
        计算超边密度
        :param rho_x:  节点密度
        :param H:       关系矩阵
        :return:
        '''
        rho_hyper_edge = torch.sum(H*rho_x,dim=0)
        return rho_hyper_edge.reshape(-1,1)
    def hyper_edge_concat(self,COE_Edge,E_tilde):
        return self.net_ELU(torch.mm(COE_Edge.T,E_tilde))

    def _prepare_attentional_mechanism_input(self, Xi,Ek):

        WX = torch.mm(Xi, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        WE = torch.mm(Ek, self.W)
        N = WX.size()[0]  # number of nodes
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
        WX_repeated_in_chunks = WX.repeat_interleave(N, dim=0)
        # repeat_interleave(self: Tensor, repeats: _int, dim: Optional[_int]=None)
        # 参数说明：
        # self: 传入的数据为tensor
        # repeats: 复制的份数
        # dim: 要复制的维度，可设定为0/1/2.....
        WE_repeated_alternating = WE.repeat(N, 1)
        # repeat方法可以对 Wh 张量中的单维度和非单维度进行复制操作，并且会真正的复制数据保存到内存中
        # repeat(N, 1)表示dim=0维度的数据复制N份，dim=1维度的数据保持不变

        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([WX_repeated_in_chunks, WE_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)


        return all_combinations_matrix.view(N, N, 2 * self.W.shape[1])

def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans =transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data",train=True,transform=trans,download=True)
    mnist_test =torchvision.datasets.FashionMNIST(
        root='../data',train=False,transform=trans,download=True
    )
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=0),
            (data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=0)))





