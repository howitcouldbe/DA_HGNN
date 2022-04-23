from DAHGNN import load_data_fashion_mnist
from DAHGNN import DA_HGNN
from torch import nn
import torch
from d2l import torch as d2l
def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()  #使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval（）时，
                    # 框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大！！！！！！
    if not device:
        device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    for X,y in data_iter:
        if isinstance(X,list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X),y),y.numel())    #numel()返回元素个数
        break
    return metric[0]/metric[1]


# def train(net,train_iter,test_iter,num_epochs,lr,device):
def train(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weights(m):
        if  type(m) == DA_HGNN:
            for parameter in m.parameters():
                nn.init.xavier_uniform_(parameter)#xavier参数初始化，使每一层的参数均值和方差都保持不变
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights) #对net里的每一个Parameter都run一下init_weights函数
    print('training on ',device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    # num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # print("=============更新之前===========")
        # temp = 0  # 控制打印的参数个数
        # for name, parms in net.named_parameters():
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        metric = d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            print(f'batch {i}')
            optimizer.zero_grad()
            X, y = X.to('cuda:0'), y.to('cuda:0')
            # y = y.to(torch.long)
            y_h = net(X)
            l = loss(y_h,y)
            l.backward()
            optimizer.step()    #进行参数优化
            with torch.no_grad():
                metric.add(l*X.shape[0],d2l.accuracy(y_h,y),X.shape[0])
        train_l = metric[0]/metric[2]
        train_acc = metric[1]/metric[2]
        test_acc = evaluate_accuracy_gpu(net,test_iter)
        print(f'loss {train_l:.3f},train acc {train_acc:.3f},test acc {test_acc:.3f}'
          f'on {str(device)}')
        # print("=============更新之后===========")
        # for name, parms in net.named_parameters():
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     print('-->grad_requirs:', parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        # print(optimizer)

train_iter,test_iter = load_data_fashion_mnist(256)
net = nn.Sequential(DA_HGNN(top_k=10,num_feature=784,sigma=0.4,multi_head=4),
                    nn.Linear(256,10),nn.Softmax()
                    )
net.to(device=torch.device('cuda:0'))
# for (X,y) in train_iter:
#
train(net,train_iter,test_iter,num_epochs=2000,lr=0.002,device='cuda:0')
    # train(net, X,y, test_iter, num_epochs=2000, lr=0.002, device='cuda:0')

# net = nn.Sequential(DA_HGNN(top_k=10,num_feature=784,batch_size=128,sigma=0.4),
#                     nn.Linear(256,10),
#                     nn.Softmax(dim=1)
#                     )
# net.to(device=torch.device('cuda:0'))
# for i,(X,y) in enumerate(train_iter):
#
#     print(net(X)[:10,:10])
#     break




