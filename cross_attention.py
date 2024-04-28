import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import random
import timm
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
from utils import *
from efficientnet_pytorch import EfficientNet
from data_loader import *
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import BertForSequenceClassification, XLNetForSequenceClassification
from models import *
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(50)

images_path = "./Dataset/images"
train_csv_path = './Dataset/train.xlsx'
test_csv_path = './Dataset/test.xlsx'

train_df = pd.read_excel(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(train_df)


rows_with_empty_values = train_df[train_df.isna().any(axis=1)]
print(rows_with_empty_values)

print(test_df)

rows_with_empty_values = test_df[test_df.isna().any(axis=1)]
print(rows_with_empty_values)

print(train_df['tweet'][0])


print(len(train_df))
print(len(test_df))


class_labels= ['ND','DI','DN','Fires','Flood','HD']



trdata = {
    'image': [],
    'tweet': [],
    'label': []
}
combinedf = pd.DataFrame(trdata)

tedata = {
    'image': [],
    'tweet': [],
    'label': []
}
testdf = pd.DataFrame(tedata)

label_map = {
    "non_damage": 0,
    "damaged_infrastructure": 1,
    "damaged_nature": 2,
    "fires": 3,
    "flood": 4,
    "human_damage": 5
}

label_map_inversed = dict((value, key) for key, value in label_map.items())

combinedf['label'] = train_df['label'].map(label_map)
testdf['label'] = test_df['label'].map(label_map)
testdf['dir'] = test_df['label'].map(label_map_inversed)

def replace_string(row):
  return row.replace('.JPG', '.jpg')


combinedf['image'] = train_df['image'].apply(replace_string)
testdf['image'] = test_df['image'].apply(replace_string)

combinedf['tweet']= train_df['tweet']
testdf['tweet']= test_df['tweet']
# combinedf['tweet'].fillna('', inplace=True)
# testdf['tweet'].fillna('', inplace=True)#box 14

rows_with_empty_values = combinedf[combinedf.isna().any(axis=1)]
print(rows_with_empty_values)

rows_with_empty_values = testdf[testdf.isna().any(axis=1)]
print(rows_with_empty_values)



combinedf = combinedf.dropna().reset_index(drop=True)
rows_with_empty_values = combinedf[combinedf.isna().any(axis=1)]
print(rows_with_empty_values)



rows_with_empty_values = combinedf[combinedf.isna().any(axis=1)]
print(rows_with_empty_values)

#combinedf = shuffle(combinedf, random_state=42)
traindf=combinedf[:4662]
validdf= combinedf[4662:]

traindf['dir'] = traindf['label'].map(label_map_inversed)
validdf['dir'] = validdf['label'].map(label_map_inversed)
testdf['dir'] = testdf['label'].map(label_map_inversed)
print(len(combinedf))
print(len(traindf))
print(len(validdf))
print(len(testdf))

print(traindf)


validdf.reset_index(drop=True, inplace=True)

print(validdf)


class_counts = traindf['label'].value_counts()
print(class_counts)

class_counts = validdf['label'].value_counts()
print(class_counts)

class_counts = testdf['label'].value_counts()
print(class_counts)

transform_train = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


transform_test = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

learning_rate = 3e-3
momentum = 0.9
beta_2 = 0.999
epsilon = None
weight_decay = 0.0
amsgrad = False


def callbacks_check(model_name):
    num_classes = 6
    accuracy_threshold = 0.99
    class MyCallback:
        def __init__(self):
            self.best_accuracy = 0.0
        def on_epoch_end(self, epoch, accuracy):
            if accuracy > accuracy_threshold:
                print("\nReached {:.2f}% accuracy, so we will stop training".format(accuracy_threshold * 100))
                return True
            return False
    acc_callback = MyCallback()
    return acc_callback


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length= 300

combined_train_dataset = CustomCombinedDataset(traindf, images_path, tokenizer, max_length, transforms=transform_train)
combined_valid_dataset = CustomCombinedDataset(validdf, images_path, tokenizer, max_length, transforms=transform_test)
combined_test_dataset = CustomCombinedDataset(testdf, images_path, tokenizer, max_length, transforms=transform_test)

len(combined_train_dataset)

criterion = nn.CrossEntropyLoss()
num_classes = 6
batch_size = 12
num_iters= 30000
num_epochs = 20#num_iters / (len(combined_train_dataset) / batch_size)
num_epochs = int(num_epochs)
print(num_epochs)

combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
combined_valid_loader = DataLoader(combined_valid_dataset, batch_size=batch_size, shuffle=False)
combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)
print(len(combined_train_loader), len(combined_test_loader), len(combined_valid_loader))
# xyz



num_classes = 6
dropout_rate= 0.2

combined_model = MultimodalClassifier_CA(num_classes=num_classes)
combined_model.to(device)

learning_rate = 1e-4
momentum = 0.9
beta_2 = 0.999
epsilon = None
weight_decay = 0.0
amsgrad = False

optimizer = optim.Adam(combined_model.parameters(),
                           lr=learning_rate)

criterion = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

patience = 3
best_val_loss = float('inf')
counter = 0

callback_list = callbacks_check(combined_model)

for epoch in range(num_epochs):
    combined_model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    print(f"Epoch {epoch}")
    for batch_idx, batch_data in enumerate(tqdm(combined_train_loader, total = len(combined_train_loader), desc = "Training")):
        
        images = batch_data['image'].to(device)
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        labels = batch_data['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = combined_model(images, input_ids, attention_mask)

        # img_output = combined_model(images, input_ids, attention_mask)
        
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    epoch_loss = total_loss / len(combined_train_loader)
    epoch_accuracy = correct / total_samples
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')
    
    combined_model.eval()
    valid_correct = 0
    valid_total_samples = 0
    valid_loss = 0.0
    
    with torch.no_grad():
        for valid_batch_data in combined_valid_loader:
            valid_images = valid_batch_data['image'].to(device)
            valid_input_ids = valid_batch_data['input_ids'].to(device)
            valid_attention_mask = valid_batch_data['attention_mask'].to(device)
            valid_labels = valid_batch_data['label'].to(device)
            
            valid_logits, _ = combined_model(valid_images, valid_input_ids, valid_attention_mask)
            valid_loss += criterion(valid_logits, valid_labels).item()
            
            _, valid_predicted = torch.max(valid_logits, 1)
            valid_correct += (valid_predicted == valid_labels).sum().item()
            valid_total_samples += valid_labels.size(0)
            
    valid_epoch_loss = valid_loss / len(combined_valid_loader)
    valid_epoch_accuracy = valid_correct / valid_total_samples
    valid_losses.append(valid_epoch_loss)
    valid_accuracies.append(valid_epoch_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_epoch_loss:.4f}, Validation Accuracy: {valid_epoch_accuracy:.4f}')
    
    if callback_list.on_epoch_end(epoch+1,epoch_accuracy):
            print(print("Early stopping triggered at epoch",epoch+1))
            break
            
    elif (valid_epoch_loss < best_val_loss):
            best_val_loss = valid_epoch_loss
            counter = 0
            torch.save(combined_model.state_dict(),'Combine_best_model_CA.pt')
            print("Combine Model saved")
            combined_model.train()
            
    else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}.')
                break

combined_model.load_state_dict(torch.load('Combine_best_model_CA.pt'))
print('Model Loaded')

ctest_labels_lst=[]
ctest_predicted_lst=[]

with torch.no_grad():
    for ctest_batch_data in combined_test_loader:
        ctest_images = ctest_batch_data['image'].to(device)
        ctest_input_ids = ctest_batch_data['input_ids'].to(device)
        ctest_attention_mask = ctest_batch_data['attention_mask'].to(device)
        ctest_labels = ctest_batch_data['label'].to(device)
        
        ctest_logits, _ = combined_model(ctest_images, ctest_input_ids, ctest_attention_mask)

        # ctest_output = combined_model(ctest_images, ctest_input_ids, ctest_attention_mask)
        
        _, ctest_predicted = torch.max(ctest_logits, 1)
        ctest_labels_lst.extend(ctest_labels.cpu().numpy()) 
        ctest_predicted_lst.extend(ctest_predicted.cpu().numpy())


ctest_labels_np = np.array(ctest_labels_lst)
ctest_predicted_np = np.array(ctest_predicted_lst)

sklearn_accuracy = accuracy_score(ctest_labels_np, ctest_predicted_np)
precision = precision_score(ctest_labels_np, ctest_predicted_np, average='weighted')
recall = recall_score(ctest_labels_np, ctest_predicted_np, average='weighted')
f1 = f1_score(ctest_labels_np, ctest_predicted_np, average='weighted')

print(f'Accuracy: {sklearn_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

class_labels= ['ND','DI','DN','Fires','Flood','HD']

true_labels = np.array(ctest_labels_lst)
predicted_labels = np.array(ctest_predicted_lst)

report = classification_report(true_labels, predicted_labels, target_names=class_labels)

print("Classification Report:")
print(report)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_confusion = confusion_matrix(ctest_labels_lst, ctest_predicted_lst)
test_confusion_display = ConfusionMatrixDisplay(confusion_matrix=test_confusion, display_labels=class_labels)
test_confusion_display.plot(cmap=plt.cm.Blues)
plt.title('Test Confusion Matrix')
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train & Validation Accuracy')

# plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train & Validation Loss')
plt.show()


test_sample = testdf.iloc[0]
image_path = os.path.join(images_path, test_sample['dir'], 'images', test_sample['image'])

image_tensor = load_and_transform_image(image_path, transform_test)
text = test_sample["tweet"]
input_ids = tokenizer.encode_plus(text, add_special_tokens=True, max_length=300, return_tensors="pt", padding="max_length", truncation=True)

#image_path, text, model, tokenizer, transform_test
visualize_cross_attention(image_path, text, combined_model, tokenizer, transform_test, device)



