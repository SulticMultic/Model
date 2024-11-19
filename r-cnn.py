import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import box_iou


# Настраиваем модель Faster R-CNN
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Датасет
class EmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        label_path = os.path.join(self.root, "labels", self.annotations[idx])

        img = Image.open(img_path).convert("RGB")
        boxes, labels = [], []

        with open(label_path) as f:
            for line in f:
                cls, x_center, y_center, width, height = map(float, line.split())
                cls = int(cls)
                image_width, image_height = img.size
                x_min = (x_center - width / 2) * image_width
                y_min = (y_center - height / 2) * image_height
                x_max = (x_center + width / 2) * image_width
                y_max = (y_center + height / 2) * image_height

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(cls)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)


# Создание датасета
dataset = EmotionsDataset("/affectnet-yolo-format/YOLO_format/train", transforms=F.to_tensor)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Инициализация модели
num_classes = 9  # 8 эмоций + 1 для фона
model = get_model(num_classes)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Оптимизатор
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


# Функция расчета точности через IoU
def evaluate_accuracy(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                pred_labels = output['labels'].cpu()
                gt_boxes = targets[i]['boxes'].cpu()

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    iou = box_iou(pred_boxes, gt_boxes)
                    correct += (iou.max(dim=0)[0] > iou_threshold).sum().item()
                total += len(gt_boxes)

    return correct / total if total > 0 else 0


# Обучение и отображение точности
num_epochs = 5
accuracy_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    accuracy = evaluate_accuracy(model, data_loader, device)
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

# Построение графика точности
plt.plot(range(1, num_epochs + 1), accuracy_history, label="Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()