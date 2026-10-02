import torch
from torchvision import transforms, datasets
import torch.utils.data as data
import os
from tqdm import tqdm
from model.vit_model import vit_base_patch16_224_in21k
import pandas as pd

def test_data_process(data_path, batch_size):
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    test_path = os.path.join(data_path, 'test')
    if not os.path.exists(test_path):
        test_path = os.path.join(data_path, 'val')

    assert os.path.exists(test_path), 'test path does not exist'

    test_dataset = datasets.ImageFolder(test_path, test_tf)
    print(f"类别: {test_dataset.classes}\n测试集: {len(test_dataset)} 张")

    test_loader = data.DataLoader(test_dataset,
                                  batch_size = batch_size,
                                  shuffle = False,
                                  num_workers = 4,
                                  )
    return test_loader

def test_model_process(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'device:{device}')

    model = model.to(device)

    test_nums = 0
    test_corrects = 0.0

    model.eval()
    with torch.no_grad():
        for b_x, b_y in tqdm(data_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            preds = torch.argmax(output, dim=1)
            test_corrects += (preds == b_y).sum().item()
            test_nums += b_x.size(0)

    test_acc = test_corrects / test_nums
    print(f'测试准确率：: {test_acc}')

if __name__ == "__main__":
    model = vit_base_patch16_224_in21k(num_classes=5, has_logits=False)

    weights_path = "weights/model.pth"
    if os.path.exists(weights_path):
        weights = torch.load(weights_path, map_location=torch.device('cpu'))
        result = model.load_state_dict(weights, strict=False)

    data_path = "./data"
    batch_size = 8

    test_loader = test_data_process(data_path, batch_size)
    test_model_process(model, test_loader)