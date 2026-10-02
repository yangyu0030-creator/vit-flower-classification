import time
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms, datasets
import torch.utils.data as Data
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt
from model.vit_model import vit_base_patch16_224_in21k

def train_val_process(data_path, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),        # 随机裁剪 + 缩放
        transforms.RandomHorizontalFlip(),        # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_dir = os.path.join(data_path, "train")
    val_dir   = os.path.join(data_path, "val")

    assert os.path.isdir(train_dir), f"找不到训练目录: {train_dir}"
    assert os.path.isdir(val_dir),   f"找不到验证目录: {val_dir}"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_tf)

    print(f"类别: {train_dataset.classes}")
    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

    return {"train": train_loader, "val": val_loader}

def train_model_process(model, data_loader, num_epochs, lr):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    criterion = nn.CrossEntropyLoss()

    # weight decay 不加在 bias 和 LayerNorm 的权重上
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": 0.05},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        model.train()
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        train_loader = data_loader["train"]
        val_loader = data_loader["val"]

        for b_x, b_y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            b_x, b_y = b_x.to(device), b_y.to(device)

            optimizer.zero_grad()
            output = model(b_x)
            loss = criterion(output, b_y)

            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=1)
            train_corrects += (preds == b_y).sum().item()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        epoch_train_loss = train_loss / train_num
        epoch_train_acc = train_corrects / train_num
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        model.eval()
        with torch.no_grad():
            for b_x, b_y in tqdm(val_loader, desc=f"Val   E{epoch}"):
                b_x, b_y = b_x.to(device), b_y.to(device)

                output = model(b_x)
                loss = criterion(output, b_y)

                preds = output.argmax(dim=1)
                val_corrects += (preds == b_y).sum().item()
                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)

        epoch_val_loss = val_loss / val_num
        epoch_val_acc = val_corrects / val_num
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"train loss={epoch_train_loss:.4f} acc={epoch_train_acc:.4f} | "
              f"val loss={epoch_val_loss:.4f} acc={epoch_val_acc:.4f}")

        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  -> best acc={best_acc:.4f}, saved")

        scheduler.step()

    time_elapsed = time.time() - since
    print(f"\n训练完成，用时 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"最佳验证准确率: {best_acc:.4f}")

    torch.save(best_model_wts, "weights/model.pth")

    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_losses,
        "val_loss_all": val_losses,
        "train_acc_all": train_accs,
        "val_acc_all": val_accs,
    })

    return model, train_process

if __name__ == "__main__":
    model = vit_base_patch16_224_in21k(num_classes=5, has_logits=False)

    weights_path = "weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
    if os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location=torch.device("cpu"))

        for k in ["head.weight", "head.bias",
                  "pre_logits.fc.weight", "pre_logits.fc.bias"]:
            weights.pop(k, None)

        result = model.load_state_dict(weights, strict=False)
        print(f"missing keys: {result.missing_keys}")
        print(f"unexpected keys: {result.unexpected_keys}")

    data_path = "./data"
    batch_size = 8
    num_epochs = 3
    lr = 1e-4

    data_loader = train_val_process(data_path=data_path, batch_size=batch_size)
    model, train_process = train_model_process(
        model=model, data_loader=data_loader,
        num_epochs=num_epochs, lr=lr,
    )

    # 保存训练过程 CSV
    train_process.to_csv("weights/train_process.csv", index=False)

    # 画曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_process["epoch"], train_process["train_loss_all"], label="train")
    axes[0].plot(train_process["epoch"], train_process["val_loss_all"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(train_process["epoch"], train_process["train_acc_all"], label="train")
    axes[1].plot(train_process["epoch"], train_process["val_acc_all"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("acc")
    axes[1].legend(); axes[1].set_title("Accuracy")

    plt.tight_layout()
    plt.savefig("weights/train_curve.png", dpi=150)
    plt.show()
