import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import gym
import numpy as np 
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cpu")


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 5, 128, 3

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.

train_data = np.vstack(np.load('train_data.npy', allow_pickle=True))
state = np.vstack(train_data[0, 0:50000]).astype(np.float)
action = np.vstack(train_data[1, 0:50000]).astype(np.float)
observation = np.vstack(train_data[2, 0:50000]).astype(np.float)
x_train = np.concatenate((state,action), axis=1)

x = torch.tensor(x_train, requires_grad=True, device=device, dtype = dtype)
y = torch.tensor(observation, requires_grad=True, device=device, dtype = dtype)

print(x.shape,y.shape)

# x = torch.randn(N, D_in, device=device)
# y = torch.randn(N, D_out, device=device)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=True),
).to(device)

# model = TwoLayerNet(D_in, H, D_out)


loss_fn = torch.nn.MSELoss(reduction='mean')

learning_rate = 1e-2

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

trainloader = DataLoader(TensorDataset(x,y), batch_size = 2000, shuffle=True)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.

######################################## online learning 1 datapoint at a time
# hold_loss=[]
# for iteration in range(x.shape[0]):
#     # Forward pass: compute predicted y by passing x to the model.
    
#     # loss = 
    
#     y_pred = model(x[iteration,:])

#     # Compute and print loss.
#     loss = loss_fn(y_pred, y[iteration,:])

#     # Before the backward pass, use the optimizer object to zero all of the
#     # gradients for the variables it will update (which are the learnable
#     # weights of the model). This is because by default, gradients are
#     # accumulated in buffers( i.e, not overwritten) whenever .backward()
#     # is called. Checkout docs of torch.autograd.backward for more details.
#     optimizer.zero_grad()

#     # Backward pass: compute gradient of the loss with respect to model
#     # parameters. Use autograd to compute the backward pass. This call will compute the
#     # gradient of loss with respect to all Tensors with requires_grad=True.
#     # After this call w1.grad and w2.grad will be Tensors holding the gradient
#     # of the loss with respect to w1 and w2 respectively.
#     loss.backward()

#     # Calling the step function on an Optimizer makes an update to its
#     # parameters
#     optimizer.step()
#     # print(loss.item())
#     hold_loss.append(loss.item())
#     print('iteration {}, loss {}'.format(iteration,loss.item()))
	    
######################################### entire dataset at the same time
# for iteration in range(1500):
#     # Forward pass: compute predicted y by passing x to the model.
    
#     # loss = 
    
#     y_pred = model(x)

#     # Compute and print loss.
#     loss = loss_fn(y_pred, y)
    
#     # print statistics
#     # running_loss += loss.item()
#     # if i % 80 == 79:    # print every 2000 mini-batches
#     #     print('[%d, %5d] loss: %.3f' %
#     #           (epoch + 1, i + 1, running_loss / 2000))
#     #     running_loss = 0.0
#     # print (running_loss)

#     # Before the backward pass, use the optimizer object to zero all of the
#     # gradients for the variables it will update (which are the learnable
#     # weights of the model). This is because by default, gradients are
#     # accumulated in buffers( i.e, not overwritten) whenever .backward()
#     # is called. Checkout docs of torch.autograd.backward for more details.
#     optimizer.zero_grad()

#     # Backward pass: compute gradient of the loss with respect to model
#     # parameters. Use autograd to compute the backward pass. This call will compute the
#     # gradient of loss with respect to all Tensors with requires_grad=True.
#     # After this call w1.grad and w2.grad will be Tensors holding the gradient
#     # of the loss with respect to w1 and w2 respectively.
#     loss.backward()

#     # Calling the step function on an Optimizer makes an update to its
#     # parameters
#     optimizer.step()
#     # print(loss.item())
#     print('iteration {}, loss {}'.format(iteration,loss.item()))


#####################################mini batch type
tol = 1e-3
running_loss = 100.0
hold_loss=[]

for epoch in range(50):
    # Forward pass: compute predicted y by passing x to the model.
    
    # print(running_loss)
    if(running_loss<tol):
        break

    batch_iter = 0

    for x_batch,y_batch in trainloader:
        batch_iter += 1
        y_pred = model(x_batch)

        # Compute and print loss.
        loss = loss_fn(y_pred, y_batch)
        
        # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0
        # print (running_loss)
        running_loss = loss.item()

        print('epoch ({},{}), loss {}'.format(epoch,batch_iter,loss.item()))

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters. Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
        hold_loss.append(running_loss)

        if(running_loss<tol):
            print("loss converged")
            break

torch.save(model.state_dict(), 'pendulum_model.pt')

plt.ylabel('MSE Loss')
plt.xlabel('epochs')
plt.title('Training Curve')

plt.plot(np.array(hold_loss))
plt.show() 

# vis = visdom.Visdom()

# loss_window = vis.line(
#     Y=torch.zeros((1)).cpu(),
#     X=torch.zeros((1)).cpu(),
#     opts=dict(xlabel='epoch',ylabel='Loss',title='training loss',legend=['Loss']))
