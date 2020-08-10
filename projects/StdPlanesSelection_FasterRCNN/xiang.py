from typing import Dict, List
import os
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision import transforms


# TODO dataset
class MyDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, category2id: Dict):
        assert os.path.exists(img_root)
        assert os.path.exists(csv_path)
        self.samples = []
        self.labels = []
        with open(csv_path, encoding='utf-8') as f:
            for line in f.readlines():
                file_name, width, height, x1, y1, x2, y2, BigClass, StdClass = line.strip().split(',')
                clsname = '{}_{}'.format(BigClass, StdClass)
                clsid = category2id[clsname]

                img_path = os.path.join(img_root, file_name)
                if os.path.exists(img_path):
                    self.labels.append(clsid)
                    self.samples.append(img_path)
        self.augements = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.augements(Image.open(self.samples[index]).convert('RGB'))
        return image, self.labels[index]

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    device = torch.device('cpu')
    batch_size = 8
    csv_path = r'/home/ultrasonic/detectron22/projects/StdPlanesSelection_FasterRCNN/datasets/huang_6planes/train.csv'
    img_root = r'/home/ultrasonic/hnumedical/SourceCode/StdPlane-2/CenterNet-master/data/Muti_Planes_Det/images'
    category2id = {'丘脑_标准': 0, '丘脑_非标准': 1, '腹部_标准': 2, '腹部_非标准': 3, '股骨_标准': 4, '股骨_非标准': 5,
                   '脊柱_标准': 6, '脊柱_非标准': 7, '小脑水平横切面_标准': 8, '小脑水平横切面_非标准': 9,
                   '四腔心切面_标准': 10, '四腔心切面_非标准': 11}
    dataset = MyDataset(csv_path, img_root, category2id)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network = resnet50(pretrained=False, num_classes=6).to(device)
    network.load_state_dict(torch.load('./1.pth'),strict=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    print(network)

    # TODO train
    epoches = 20
    network.train()
    for epoch in range(1, epoches, 1):
        print('Epoch : {} ....'.format(epoch))

        running_loss = 0.0
        running_acc = 0.0
        steps = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += torch.sum(preds == labels.data)
            steps += 1
            if steps % 20 == 0:
                print('    steps={} ,loss={} , acc={}'.format(steps, running_loss / 20,
                                                              running_acc / (batch_size*20)))
                running_loss = 0.
                running_acc = 0.

        print('Saving {}.pth'.format(epoch))
        torch.save(network.state_dict(), '/data/will/StdPlanesSelection_FasterRCNN/logs/cls/{}.pth'.format(epoch))
