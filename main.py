import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models
import pandas as pd
import os
from torch.nn import InstanceNorm1d
from torchvision.models import vgg19
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import json
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="1" 


class CustomDataset(Dataset):
    def __init__(self, tsv_file, image_dir, json_file, tokenizer, transform=None, max_samples=5000):
        # 仅读取前5000行数据
        self.data = pd.read_csv(tsv_file, delimiter='\t', nrows=max_samples)
        self.image_dir = image_dir
        self.json_data = None
        with open(json_file, 'r') as file:
            self.json_data = json.load(file)
        self.tokenizer = tokenizer
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        while True:
            try:
                if idx >= len(self.data):
                    raise IndexError("Index out of bounds.")
                
                row = self.data.iloc[idx]
                image_name = row['id']
                image_path = os.path.join(self.image_dir, image_name + '.jpg')
                image = Image.open(image_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                text = row['title']
                inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
                input_ids = inputs['input_ids'][0]
                attention_mask = inputs['attention_mask'][0]

                json_entry = next((item for item in self.json_data if item["image_name"] == image_name), None)
                if json_entry is None:
                    idx = (idx + 1) % len(self.data)
                    continue
                
                json_features = json_entry['description']
                other_inputs = self.tokenizer(json_features, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
                other_input_ids = other_inputs['input_ids'][0]
                other_attention_mask = other_inputs['attention_mask'][0]

                label = torch.tensor(row['2_way_label'], dtype=torch.long)
                
                return {'image': image, 'input_ids': input_ids, 'attention_mask': attention_mask, 
                        'other_input_ids': other_input_ids, 'other_attention_mask': other_attention_mask, 'label': label}
            
            except IndexError:
                raise IndexError("Index out of bounds.")
            except Exception as e:
                idx = (idx + 1) % len(self.data)
                continue

class FCModule(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5, dense_units=256):
        super(FCModule, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(input_dim, dense_units)
        self.norm = nn.LayerNorm(dense_units)
        self.out = nn.Linear(dense_units, 2)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.functional.relu(x)
        x = self.norm(x)
        x = self.out(x)
        return x

class ParametricGatedWeightModule(nn.Module):
    def __init__(self, W_p):
        super(ParametricGatedWeightModule, self).__init__()
        self.W_p = W_p

    def forward(self, p):
        max_val = p.matmul(self.W_p).squeeze()
        max_val_clamped = torch.clamp(max_val, min=0)
        return max_val_clamped

class CoAttentionTransformer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CoAttentionTransformer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)


        self.Wo = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, I1, I2):
        """
        I1: First modality input, shape [batch_size, seq_len_1, d_model]
        I2: Second modality input, shape [batch_size, seq_len_2, d_model]
        """
        batch_size = I1.size(0)

        Q = self.Wq(I1).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L1, D/H]
        K = self.Wk(I2).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L2, D/H]
        V = self.Wv(I2).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L2, D/H]

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, H, L1, L2]
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B, H, L1, L2]

        attended_values = torch.matmul(attn_probs, V)  # [B, H, L1, D/H]
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [B, L1, D]

        H = self.Wo(attended_values)  # [B, L1, D]


        H = self.norm1(I1 + H)  # [B, L1, D]

        H_prime = self.norm2(H + self.ffn(H))  # [B, L1, D]


        F = H_prime.mean(dim=1)  # [B, D]

        return F

class AttentionFusion(nn.Module):
    def __init__(self, image_dim, text_dim):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(image_dim, image_dim)
        self.key = nn.Linear(text_dim, text_dim)
        self.value = nn.Linear(text_dim, text_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key_value):
        """
        query: 图像特征或文本特征 [batch_size, feature_dim]
        key_value: 另一个模态的特征 [batch_size, feature_dim]
        """
        query = self.query(query)
        key = self.key(key_value)
        value = self.value(key_value)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attn_weights = self.softmax(attn_scores)
        

        attended_features = torch.matmul(attn_weights, value)
        
        return attended_features, attn_weights

class MultiModalModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiModalModel, self).__init__()
        self.vgg19 = vgg19(pretrained=True)
        self.vgg19.classifier = nn.Sequential(*list(self.vgg19.classifier.children())[:-1]) 
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Identity()  

        self.bert = BertModel.from_pretrained('pretraining/bert-base-uncased')


        self.image_feature_extractor_vgg = nn.Sequential( 
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.image_feature_extractor_resnet = nn.Sequential(
            nn.Linear(512, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.local_feature_processor = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.text_feature_extractor = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.other_text_feature_extractor = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.weight_generator = nn.Sequential(
            nn.Linear(256 , 512),  
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),  
            nn.Softmax(dim=1)  
        )

        self.Attention_text=AttentionFusion(256,256)
        self.Attention_image=AttentionFusion(256,256)
        self.Attention_image_1=AttentionFusion(256,256)
        self.Attention_image_local=AttentionFusion(256,256)


        self.vgg_adjuster = nn.Linear(4096, 256)  
        self.resnet_adjuster = nn.Linear(512, 256)  
        self.ct = CoAttentionTransformer(d_model=256, n_heads=8, dropout=0.1)


        #self.object_detector = fasterrcnn_resnet50_fpn(pretrained=True)        


        self.image_weight = nn.Parameter(torch.tensor(1.0))
        self.text_weight = nn.Parameter(torch.tensor(1.0))
        self.other_text_weight = nn.Parameter(torch.tensor(1.0))

        #self.W_p = nn.Parameter(torch.Tensor(256,256))

        #nn.init.xavier_uniform_(self.W_p)
        #self.parametric_gated_weight_module = ParametricGatedWeightModule(self.W_p)


        self.classifier_1 = nn.Sequential(            
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)) 
        self.classifier_2 = nn.Sequential(            
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)) 
        self.classifier_3 = nn.Sequential(            
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes))      
        self.classifier_4 = nn.Sequential(            
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes))       

        self.fc_layers = nn.Sequential(
            nn.Linear(256*2,1024),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes) 
        )
        self.pre_fc_features = None

    def _get_resnet_features(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        local_features = self.resnet18.layer4(x)  
        global_features = self.resnet18.avgpool(local_features)
        global_features = torch.flatten(global_features, 1)  

        return local_features, global_features
    def extract_features(self, image_tensor):

        with torch.no_grad():
            vgg_features = self.vgg19(image_tensor)
            resnet_features = self.resnet18(image_tensor)

        vgg_features = vgg_features.view(vgg_features.size(0), -1, 1, 1).squeeze(-1).squeeze(-1) 
        resnet_features = resnet_features.view(resnet_features.size(0), -1, 1, 1).squeeze(-1).squeeze(-1) 

        vgg_features = self.vgg_adjuster(vgg_features)
        resnet_features = self.resnet_adjuster(resnet_features)

        seq_len=100
        vgg_features = vgg_features.unsqueeze(1).repeat(1, seq_len, 1)  
        resnet_features = resnet_features.unsqueeze(1).repeat(1, seq_len, 1)  

        return vgg_features, resnet_features
    
    def forward(self, image, input_ids, attention_mask, other_input_ids, other_attention_mask, labels=None):
        #image_features_vgg = self.vgg19(image)
        #image_features_vgg = self.image_feature_extractor_vgg(image_features_vgg)
        image_features_resnet = self.resnet18(image)
        image_features_resnet = self.image_feature_extractor_resnet(image_features_resnet)

        vgg_features, resnet_features= self.extract_features(image)
        vgg_resnet_feature=self.ct(vgg_features, resnet_features)#resnet_features vgg_features*2 
        #print(image_features.shape)

        local_features,_ = self._get_resnet_features(image)
        local_features = self.local_feature_processor(local_features)
        local_features = local_features.view(local_features.size(0), -1)  # 扁平化


        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_features = bert_outputs.last_hidden_state[:, 0, :]  # CLS token
        bert_features = self.text_feature_extractor(bert_features)
        

        other_bert_outputs = self.bert(input_ids=other_input_ids, attention_mask=other_attention_mask)
        other_bert_features = other_bert_outputs.last_hidden_state[:, 0, :]  # CLS token
        other_bert_features = self.other_text_feature_extractor(other_bert_features)
        
        #print("------",image_features, bert_features, other_bert_features)
        attention_image_features, _ = self.Attention_image_1( image_features_resnet,bert_features)
        attention_local_features, _ = self.Attention_image( local_features,bert_features)
        attention_text_other_features, _ = self.Attention_text(bert_features, other_bert_features)
        attention_ii_features, _ = self.Attention_image_local(local_features,other_bert_features)
        #attention_tbt_features, _ = self.Attention_text(attention_text_other_features,other_bert_features )
        
        #print(attention_image_features.shape)


        cos_sim=torch.nn.CosineSimilarity(dim=0,eps=1e-6)
        sim_text_other=cos_sim(bert_features,other_bert_features)
        sim_image_other=cos_sim(image_features_resnet,other_bert_features)
        sim_image_text=cos_sim(image_features_resnet,bert_features)

        total_sim_t=torch.sum(sim_text_other)
        total_sim_i=torch.sum(sim_image_other)
        total_sim_it=torch.sum(sim_image_text)
        alpha=sim_text_other/total_sim_t
        beta=sim_image_other/total_sim_i
        gama=sim_image_text/total_sim_it

        weight_feature_image=image_features_resnet
        weight_feature_text=bert_features
        weight_feature_other=other_bert_features

        weight_feature_1=alpha*attention_text_other_features+bert_features#
        weight_feature_2=bert_features+other_bert_features#alpha*attention_tbt_features 
        weight_feature_3=beta*attention_image_features+bert_features#other_   
        weight_feature_4=beta*vgg_resnet_feature+bert_features#    
        weight_feature_5=beta*local_features+bert_features#beta*local_features+bert_features 
        weight_feature_6=beta*attention_ii_features+bert_features 
        weight_feature_7=beta*attention_local_features+bert_features
        weight=self.weight_generator(other_bert_features)
        weight_feature_8=other_bert_features*weight+bert_features

        #combined_features=torch.cat((weight_feature_1,weight_feature_4,weight_feature_6),dim=1)#
        #combined_features=bert_features

        logits_1 = self.classifier_1(weight_feature_4)
        logits_2 = self.classifier_2(weight_feature_6)
        logits_3 = self.classifier_3(weight_feature_1)
        logits_4 = self.classifier_4(weight_feature_1)
        
        #labels=None

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            losses = [
                loss_fn(logits_1, labels),
                loss_fn(logits_2, labels),
                #loss_fn(logits_3, labels),
                loss_fn(logits_4, labels)
            ]
            weight=0
            with torch.no_grad():
                accuracies_1 = [torch.mean((logits.argmax(dim=1) == labels).float()) for logits in [logits_1, logits_2]]
                weights_1 = torch.tensor(accuracies_1, device=logits_1.device)
                weight=weights_1
                weights_1 /= weights_1.sum()  # 归一化

            combined_features_1 = (weights_1[0] * weight_feature_4 +
                                 weights_1[1] * weight_feature_6)            
            # 加权组合特征
            combined_features_2 = weight_feature_1 
            combined_features=torch.cat((combined_features_1,combined_features_2),dim=1)#weight_feature_4

        
        #for i in range(len(self.fc_layers) - 1):
            #combined_features = self.fc_layers[i](combined_features) if i == 0 else self.fc_layers[i](combined_features)

        #self.pre_fc_features = combined_features

        #output = self.fc_layers[-1](self.pre_fc_features)
        #combined_features=weight_feature_8
        output = self.fc_layers(combined_features)
        
        return output,weight,self.pre_fc_features
    

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.fc = nn.Linear(feature_dim, 1)
    
    def forward(self, x):
        attn_scores = self.fc(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_x = torch.sum(attn_weights * x, dim=1)
        return weighted_x

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10, scheduler=None):
    model.to(device)
    best_val_acc = 0.0
    total_samples = 0
    correct_predictions = 0
    best_val_precision=0
    best_val_recall=0
    best_val_f1=0
    pre_fc_features=0
    predicted=0

    weight=0
    weights_history_a = []
    weights_history_b = []
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0 
        total_samples=0
        batch_a=[]
        batch_b=[]
        
        for batch in train_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            other_input_ids = batch['other_input_ids'].to(device)
            other_attention_mask = batch['other_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs,weight, pre_fc_features = model(images, input_ids, attention_mask, other_input_ids, other_attention_mask,labels)

            # Compute the classification loss
            class_loss = criterion(outputs, labels)
            
            # Compute the contrastive loss
            #batch_size = images.size(0)
            #contrastive_labels = torch.ones(batch_size).to(device)  # Assuming paired data
            #contrastive_loss_value_1 = contrastive_loss(torch.cat((image_features, text_features), dim=1), contrastive_labels)
            #contrastive_loss_value_2 = contrastive_loss(torch.cat((text_features, other_features), dim=1), contrastive_labels)
            #contrastive_loss_value_3 = contrastive_loss(torch.cat((image_features, other_features), dim=1), contrastive_labels)
            

            total_loss = class_loss 
            total_loss.backward()
            optimizer.step()
            
            running_loss += class_loss.item()
            

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            batch_a.append(weight[0].detach().cpu().numpy())
            batch_b.append(weight[1].detach().cpu().numpy())

        #weights_history_a.extend(batch_a)
        #weights_history_b.extend(batch_b)        

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, test_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{epochs}], Test Loss: {val_loss:.4f}, Test Acc: {val_accuracy:.4f}')
        print(f'Negative Precision: {val_precision[0]:.4f}, Negative Recall: {val_recall[0]:.4f}, Negative F1 Score: {val_f1[0]:.4f}')
        print(f'Positive Precision: {val_precision[1]:.4f}, Positive Recall: {val_recall[1]:.4f}, Positive F1 Score: {val_f1[1]:.4f}')
        #print(weight)

        if scheduler is not None:
            scheduler.step(val_loss)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_precision=val_precision
            best_val_recall=val_recall
            best_val_f1=val_f1
            torch.save(model.state_dict(), "best_model.pth")
        print("****best_acc",best_val_acc,best_val_precision,best_val_recall,best_val_f1)


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            other_input_ids = batch['other_input_ids'].to(device)
            other_attention_mask = batch['other_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs,weight,pc_feature= model(images, input_ids, attention_mask, other_input_ids, other_attention_mask,labels)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)
    
    avg_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1_score

def main(train_tsv_path, test_tsv_path, train_image_dir, test_image_dir, train_json_dir, test_json_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading----------------------------")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('pretraining/bert-base-uncased')
    print("Data Loading-----------------------------")
    train_dataset = CustomDataset(train_tsv_path, train_image_dir,train_json_dir, tokenizer, transform=transform, max_samples=5000)
    test_dataset = CustomDataset(test_tsv_path, test_image_dir, test_json_dir, tokenizer, transform=transform, max_samples=500)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
   

    print("Model loading--------------------------------------")
    model = MultiModalModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    print("Training------------------------------------------")

    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=50,scheduler=scheduler)

    #print("Evaluating--------------------------------------")
    evaluate_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    train_tsv_path = 'data/Fakeddit-master/multimodal_only_samples/multimodal_validate.tsv'
    test_tsv_path = 'data/Fakeddit-master/multimodal_only_samples/multimodal_test_public.tsv'

    train_image_dir = 'data/Fakeddit-master/multimodal_only_samples/validate'
    test_image_dir = 'data/Fakeddit-master/multimodal_only_samples/test'

    train_json_dir='data/Fakeddit-master/multimodal_only_samples/qwen_json/validate_llm_result.json'
    test_json_dir='data/Fakeddit-master/multimodal_only_samples/qwen_json/test_llm_result.json'

    main(train_tsv_path, test_tsv_path, train_image_dir, test_image_dir, train_json_dir, test_json_dir)