import torch
import torch.nn as nn
import torch.optim as optim

from train_cnn_mnist import CNN, CNNTrain

# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
class CW_Attack():
    def cw_l2_attack(self, model, images, labels, use_cuda=False, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01) :
        device = torch.device("cuda" if use_cuda else "cpu")
        print(device)

        images = images.to(device)     
        labels = labels.to(device)

        # Define f-function
        def f(x) :

            outputs = model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.byte())
            
            # If targeted, optimize for making the other class most likely 
            if targeted :
                return torch.clamp(i-j, min=-kappa)
            
            # If untargeted, optimize for making the other class most likely 
            else :
                return torch.clamp(j-i, min=-kappa)
        
        w = torch.zeros_like(images, requires_grad=True).to(device)

        optimizer = optim.Adam([w], lr=learning_rate)

        prev = 1e10
        
        for step in range(max_iter) :

            a = 1/2*(nn.Tanh()(w) + 1)

            loss1 = nn.MSELoss(reduction='sum')(a, images)
            loss2 = torch.sum(c*f(a))

            cost = loss1 + loss2

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Early Stop when loss does not converge.
            if step % (max_iter//10) == 0 :
                if cost > prev :
                    print('Attack Stopped due to CONVERGENCE....')
                    return a
                prev = cost
            
            print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

        attack_images = 1/2*(nn.Tanh()(w) + 1)

        return attack_images

# trainer = CNNTrain()
# trainer.model_train()

# model = trainer.model
# images, labels = trainer.mnist.data_test[0]
# image_torchs = images.data.view(1, 1, 28, 28).float()
# label_torchs = torch.tensor(labels)


# ae = cw_l2_attack(model, image_torchs, label_torchs)