from dataloader import read_bci_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import os
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*epoch parameter in `scheduler.step.*",
    category=UserWarning
)

def Dataset_Loader(data, label, Shuffle):
    X = torch.from_numpy(data).float()
    y = torch.from_numpy(label).long().view(-1)  
    return DataLoader(TensorDataset(X, y), batch_size=256, shuffle=Shuffle)

class EEGNet(nn.Module):
    def __init__(self, activation):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.flatten  = nn.Flatten()
        self.classify = nn.LazyLinear(2, bias=True)

    def forward(self, input):
        output = self.firstconv(input)
        output = self.depthwiseConv(output)
        output = self.separableConv(output)
        output = self.flatten(output)
        output = self.classify(output)
        return output

class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.flatten  = nn.Flatten()
        self.classify = nn.LazyLinear(2, bias=True)

    def forward(self, input):
        output = self.conv0(input)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flatten(output)
        output = self.classify(output)
        return output

def run(model, train_loader, test_loader, device, epochs=300, lr=1e-3, warmup_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    # --- Warmup + Cosine ---
    warmup_epochs = min(warmup_epochs, max(0, epochs-1))  
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)  
    T_cos = max(1, epochs - warmup_epochs)                
    cosine = CosineAnnealingLR(optimizer, T_max=T_cos, eta_min=lr * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    train_acc_hist, test_acc_hist, train_loss_hist = [], [], []
    best_acc, best_model = 0.0, None

    model.to(device)
    with tqdm(range(epochs), leave=False) as pbar:
        for ep in pbar:
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)

            train_loss = running_loss / max(total, 1)
            train_acc  = 100.0 * correct / max(total, 1)

            # ---- test ----
            model.eval()
            correct_test, total_test = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    logits = model(x)
                    correct_test += (logits.argmax(1) == y).sum().item()
                    total_test += x.size(0)
            test_acc = 100.0 * correct_test / max(total_test, 1)

            train_loss_hist.append(train_loss)
            train_acc_hist.append(train_acc)
            test_acc_hist.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)

            pbar.set_description(
                f'lr:{optimizer.param_groups[0]["lr"]:.2e}  loss:{train_loss:.4f}  '
                f'train:{train_acc:.2f}%  test:{test_acc:.2f}%  best:{best_acc:.2f}%'
            )
            scheduler.step()

    return train_acc_hist, test_acc_hist, train_loss_hist, best_model, best_acc

def show_result(model_name, acc):
    plt.figure(figsize=(10, 6))
    plt.title(f'Activation Function Comparision({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    for name, curve in acc.items():
        plt.plot(curve, label = name)
        print(name + ' Max Accuracy:' + str(max(curve)))
    plt.legend()
    plt.savefig(f'{model_name}.png')
    plt.close()

def show_loss_curve(model_name, loss_curve):
    plt.figure(figsize=(10, 6))
    plt.title(f'Loss Curve({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for name, curve in loss_curve.items():
        plt.plot(curve, label = name)
    plt.legend()
    plt.savefig(f'{model_name}_loss_curve.png')
    plt.close()

def print_activation_summary(model_name, acc, best_score):

    rows = []
    for act in sorted(best_score.keys()):
        tr_max = max(acc[f"Train_{act}"]) if acc.get(f"Train_{act}") else float("nan")
        te_max = max(acc[f"Test_{act}"])  if acc.get(f"Test_{act}")  else float("nan")
        rows.append((act, tr_max, te_max, best_score[act]))

    rows.sort(key=lambda r: r[2], reverse=True)

    title = f"{model_name} — Activation Accuracy Summary"
    print("\n" + title)
    print("-" * len(title))
    print(f"{'Activation':<12}{'Train Max (%)':>14}{'Test Max (%)':>14}{'Best Saved (%)':>16}")
    print("-" * (12 + 14 + 14 + 16))
    for act, tr, te, best in rows:
        print(f"{act:<12}{tr:>14.2f}{te:>14.2f}{best:>16.2f}")

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = read_bci_data()
    train_loader = Dataset_Loader(train_data, train_label, True)
    test_loader  = Dataset_Loader(test_data,  test_label,  False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    selected_model = int(input("Select model: [1] EEGNet  /  [2] DeepConvNet  -> "))

    if selected_model == 1:
        model_name = 'EEGNet'
        builders = {
            'ELU':       lambda: EEGNet(nn.ELU()),
            'ReLU':      lambda: EEGNet(nn.ReLU()),
            'LeakyReLU': lambda: EEGNet(nn.LeakyReLU()),
        }
    else:
        model_name = 'DeepConvNet'
        builders = {
            'ELU':       lambda: DeepConvNet(nn.ELU()),
            'ReLU':      lambda: DeepConvNet(nn.ReLU()),
            'LeakyReLU': lambda: DeepConvNet(nn.LeakyReLU()),
        }

    acc = {
        'Train_ELU': None, 'Train_ReLU': None, 'Train_LeakyReLU': None,
        'Test_ELU':  None, 'Test_ReLU':  None, 'Test_LeakyReLU':  None
    }
    loss_curve = {'ELU': None, 'ReLU': None, 'LeakyReLU': None}
    best_param = {'ELU': None, 'ReLU': None, 'LeakyReLU': None}
    best_score = {'ELU': 0.0, 'ReLU': 0.0, 'LeakyReLU': 0.0}

    for act_name, make_model in builders.items():
        print(f'==> {model_name}_{act_name} <==')
        model = make_model().to(device)      
        tr_acc, te_acc, tr_loss, best_model, best_acc = run(
            model, train_loader, test_loader, device, epochs=300, lr=2e-3   # EEGNet lr=1.824e-3 、 DeepConvNet lr=2e-3
        )
        acc['Train_' + act_name] = tr_acc
        acc['Test_'  + act_name] = te_acc
        loss_curve[act_name] = tr_loss
        best_param[act_name] = best_model
        best_score[act_name] = best_acc

    os.makedirs('best', exist_ok=True)
    for act_name, model in best_param.items():
        if model is None: continue
        tag = f'{model_name}_{act_name}_{round(best_score[act_name], 2)}%.pth'
        torch.save(model, os.path.join('best', tag))

    show_result(model_name, acc)
    show_loss_curve(model_name, loss_curve)

    print_activation_summary(model_name, acc, best_score)

    print("\nSaved checkpoints (evaluated on test set)")
    print("-------------------------------------------------")
    for file in sorted(os.listdir('best')):
        if not file.endswith('.pth'): 
            continue
        name = file.rsplit('.', 1)[0]
        model = torch.load(os.path.join('best', file), map_location=device, weights_only=False)
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in test_loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        final_acc = 100.0 * correct / max(total, 1)
        print(f"{name:<32} -> {final_acc:6.2f}%")
