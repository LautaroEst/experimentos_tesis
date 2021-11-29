import torch
from torch import nn, optim


def main():
    model = nn.Sequential(nn.Linear(3,4),nn.Linear(4,1))
    model[0].weight.requires_grad = False

    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for e in range(3):
        x = torch.randn((16,3))
        y = torch.randint(0,1,(16,1),dtype=torch.float)
        
        y_hat = model(x)
        loss = criterion(y_hat,y)
        
        print(model[0].weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




if __name__ == "__main__":
    main()