import torch
from torch.utils.data import Dataset
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence

import random
import math
import heapq
import torch.fft
from tqdm import tqdm
from datasets import DatasetDict
from transformers import BertTokenizer, BertModel
import warnings
import pickle as pkl
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

def bert_embeddings(text):
  bert.eval().to(device)
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [1] * len(indexed_tokens)

  seg_vecs = []
  window_length, start = 510, 0
  loop = True
  while loop:
    end = start + window_length
    if end >= len(tokenized_text):
        loop = False
        end = len(tokenized_text)

    indexed_tokens_chunk = indexed_tokens[start : end]
    segments_ids_chunk = segments_ids[start : end]

    indexed_tokens_chunk = [101] + indexed_tokens_chunk + [102]
    segments_ids_chunk = [1] + segments_ids_chunk + [1]

    tokens_tensor = torch.tensor([indexed_tokens_chunk]).to(device)
    segments_tensors = torch.tensor([segments_ids_chunk]).to(device)
    # Hidden embeddings: [n_layers, n_batches, n_tokens, n_features]
    with torch.no_grad():
      outputs = bert(tokens_tensor, segments_tensors)
      hidden_states = outputs[2]

    seg_vecs.append(hidden_states[-2][0])
    start += window_length

  token_vecs = torch.cat(seg_vecs, dim=0)
  sentence_embedding = torch.mean(token_vecs, dim=0).cpu()
  return sentence_embedding

def haversine_distance(coord1, coord2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    # Compute differences in latitude, longitude and altitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate distance using Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000

def AnomalyDetectionDatasetSplit(data_path, abnormal_samples, normal_samples, llm_label = True):
    with open(data_path, 'rb') as f:
        datasets = pkl.load(f)

    periods = get_freq(datasets)
    train_dataset, test_dataset = list(), list()
    scores = [datasets[i]["anomaly_score"] for i in range(len(datasets))]

    if llm_label: 
        threld = heapq.nlargest(abnormal_samples, scores)[-1]
        normal_ids = [datasets[i]["id"] for i in range(len(datasets)) if datasets[i]["llm_label"] == "normal"]
        abnormal_sample_ids = [datasets[i]["id"] for i in range(len(datasets)) if scores[i] >= threld][:abnormal_samples]
        normal_sample_ids = random.sample(normal_ids, normal_samples)
    else: 
        normal_ids = [datasets[i]["id"] for i in range(len(datasets)) if datasets[i]["label"] == "normal"]
        abnormal_sample_ids = [datasets[i]["id"] for i in range(len(datasets)) if datasets[i]["label"] == "abnormal"]
        normal_sample_ids = random.sample(normal_ids, normal_samples)

    train_dataset = [datasets[i] for i in range(len(datasets)) if datasets[i]["id"] in abnormal_sample_ids or datasets[i]["id"] in normal_sample_ids]
    test_dataset = [datasets[i] for i in range(len(datasets)) if datasets[i]["id"] not in abnormal_sample_ids and datasets[i]["id"] not in normal_sample_ids]

    return  DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        }), periods

def ClassificationDatasetSplit(data_path):
    with open(data_path, 'rb') as f:
        datasets = pkl.load(f)

    periods = get_freq(datasets)
    train_dataset, test_dataset = list(), list()

    label_dic = list(set([item["label"] for item in datasets]))

    for l in label_dic: 
        subset = [item for item in datasets if item["label"] == l]
        train_dataset += random.sample(subset, int(0.7*len(subset)))
    
    test_dataset = [sample for sample in datasets if sample not in train_dataset]

    return  DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        }), periods

class TrajectoryDataset(Dataset):
    #HybridEmbeddings: Spatio-temporal only, Text Trajectory only, Fix Trajectory, controled by para: alpha
    def __init__(self, data, periods):
        self.data = data
        self.periods = periods

    def st_embed(self, st_info):
        coors = [st[0] for st in st_info]
        times = [st[1] for st in st_info]
        velocities, accelerations = [], []

        for i in range(len(times)):
            if i == 0: velocities.append(0)
            else: velocities.append(self.get_v(coors[i-1], coors[i], times[i-1], times[i]))

        for i in range(len(times)):
            if i == 0: accelerations.append(0)
            else: accelerations.append(self.get_a(velocities[i-1], velocities[i], times[i-1], times[i]))

        return torch.stack([torch.tensor([coors[i][0], coors[i][1], velocities[i], accelerations[i]], dtype = torch.float) for i in range(len(times))])

    def get_v(self, c1, c2, t1, t2):
        FMT = '%H:%M:%S'
        try:
            d_t = (datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)).total_seconds()
        except:
            d_t = t2-t1
        d = haversine_distance(c1, c2)
        if d_t != 0: v = d/d_t
        else: v = 0
        return v

    def get_a(self, v1, v2, t1, t2):
        FMT = '%H:%M:%S'
        try:
            d_t = (datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)).total_seconds()
        except:
            d_t = t2-t1
        d_v = abs(v2-v1)
        if d_t != 0: a = d_v/d_t
        else: a = 0
        return a

    def get_attn_map(self, llm_attn):
        attn_map = torch.zeros(self.periods)
        if llm_attn == None: return attn_map
        if len(llm_attn)!=0:
            for i in llm_attn:
                if i < self.periods: attn_map[i] = 1 
        return attn_map

    def text_embed(self, text_info):
        text_embedding = [bert_embeddings(text_info[i]) for i in range(len(text_info))]
        return torch.stack(text_embedding)

    def label2vec(self, text_label):
        labels = list(set([self.data[idx]["label"] for idx in range(len(self.data))]))
        return torch.tensor([labels.index(text_label)], dtype = torch.int64)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        #User id:
        userid = torch.tensor(self.data[idx]["id"], dtype = torch.int64)
        #salient segments of tokens from llms
        llm_attn = self.get_attn_map(self.data[idx]["llm_attn"])
        #Textual label
        #Groud truth label
        label = self.label2vec(self.data[idx]["label"])
        #Spatio-temporal trajectory
        st_rep = self.st_embed(self.data[idx]["st_sequence"])
        #Text description of trajectory:
        text_features = self.data[idx]["text_sequence"]
        #Label by LLMs:
        llm_label = self.data[idx]["llm_label"]
        #Anomaly score
        anomaly_score = self.data[idx]["anomaly_score"]
        try:
            anomaly_score = torch.tensor(anomaly_score, dtype=float)
        except:
            anomaly_score = None

        if torch.is_tensor(text_features): text_rep = text_features
        elif isinstance(text_features, list): text_rep = self.text_embed(text_features)
        else: raise ValueError
        return {
            "userid": userid,
            "llm_attn": llm_attn,
            "st_embedding": st_rep, 
            "text_embedding": text_rep, 
            "label": label,
            "llm_label": llm_label,
            "anomaly_score":anomaly_score
            }

def FFT_for_Period(x, k):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = torch.where(top_list == 0, torch.tensor(1), top_list)
    period = x.shape[1] // top_list

    return period[1:]

def get_freq(data, k=2):
    periods, most_freqs = [], []
    C = len(data[0]["st_sequence"][0][0])
    for d in data:
        st_sequence = d["st_sequence"]
        st_stack = [torch.tensor([st_sequence[i][0][c] for i in range(len(st_sequence))], dtype = torch.float) for c in range(C)]
        ts = torch.stack(st_stack)
        period = FFT_for_Period(ts.permute(1, 0).unsqueeze(0).contiguous(), k=k)
        periods.append(period)

    freqs = torch.stack(periods).permute(1, 0)
    for i in range(freqs.shape[0]):
        vec = freqs[i]
        unique, counts = torch.unique(vec, return_counts=True)
        frequencies = dict(zip(unique.tolist(), counts.tolist()))
        for f in range(15): frequencies.pop(f, None)
        most_freqs.append(max(frequencies, key=frequencies.get))
    return most_freqs[0]

def collate_fn(batch):
    ids = [batch[i]["userid"] for i in range(len(batch))]
    llm_attns = [batch[i]["llm_attn"] for i in range(len(batch))]
    label_list = [batch[i]["label"] for i in range(len(batch))]
    st_embeddings = [batch[i]["st_embedding"] for i in range(len(batch))]
    text_embeddings = [batch[i]["text_embedding"] for i in range(len(batch))]

    #ids, llm_attns, st_embeddings, text_embeddings, label_list = zip(*batch)
    ids = torch.stack(ids)
    llm_attns = torch.stack(llm_attns)
    st_embeddings = pad_sequence(st_embeddings, batch_first=True, padding_value=0)
    text_embeddings = pad_sequence(text_embeddings, batch_first=True, padding_value=0)
    label_list = torch.stack(label_list)
    
    return ids.to(device), llm_attns.to(device), st_embeddings.to(device), text_embeddings.to(device), label_list.to(device)

def attn_align(attention_maps, llm_attns):
    for i in range(llm_attns.shape[0]):
        if llm_attns[i].sum() == 0: llm_attns[i] = attention_maps[i]
    return llm_attns

# Training function.
def train(model, train_loader, optimizer, label_criterion, attn_criterion, return_attention):
    model.train()
    print('Training')
    train_running_loss = 0.0
    total_test_samples = 0
    correct_predictions = 0
    counter = 0
    for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        _, llm_attns, st_embeddings, text_embeddings, labels = data
        # Forward pass.
        if return_attention:
            outputs, attention_maps = model(st_embeddings, text_embeddings)
        else:
            outputs = model(st_embeddings, text_embeddings)
        outputs = torch.squeeze(outputs, -1)
        #Record predictions
        predictions = torch.argmax(outputs, dim=1).view(-1)
        total_test_samples += labels.shape[0]
        correct_predictions += (predictions == labels.view(-1)).sum().item()
        # Calculate the loss.
        if return_attention:
            loss = label_criterion(outputs, labels.view(-1)) + attn_criterion(attention_maps, attn_align(attention_maps, llm_attns))
        else:
            loss = label_criterion(outputs, labels.view(-1))
        # Backpropagation.
        loss.backward()
        # Update the optimizer parameters.
        optimizer.step()

        train_running_loss += loss.item()

    #Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    accuracy = correct_predictions / total_test_samples
    return epoch_loss, accuracy

# Validation function.
def validate(model, test_loader, label_criterion, attn_criterion, return_attention):
    model.eval()
    print('Validation')
    probs = {}
    valid_running_loss = 0.0
    total_test_samples = 0
    correct_predictions = 0
    counter = 0  
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            counter += 1
            ids, llm_attns, st_embeddings, text_embeddings, labels = data
            # Forward pass.
            if return_attention:
                outputs, attention_maps = model(st_embeddings, text_embeddings)
            else:
                outputs = model(st_embeddings, text_embeddings)
            for i in range(ids.shape[0]):
                if ids[i] in probs: probs[ids[i]] = outputs[i]
                else: probs[ids[i]] = outputs[i]
            outputs = torch.squeeze(outputs, -1)
            #Record predictions
            predictions = torch.argmax(outputs, dim=1).view(-1)
            total_test_samples += labels.shape[0]
            correct_predictions += (predictions == labels.view(-1)).sum().item()
            # Calculate the loss.
            if return_attention:
                loss = label_criterion(outputs, labels.view(-1)) + attn_criterion(attention_maps, attn_align(attention_maps, llm_attns))
            else:
                loss = label_criterion(outputs, labels.view(-1))
            # loss.item() * labels.shape[0]
            valid_running_loss += loss.item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    accuracy = correct_predictions / total_test_samples
    return epoch_loss, accuracy, probs

def topk_hits(k, scores, user_ids, truth_ids):
    threhold = heapq.nlargest(k, scores)[-1]
    count = 0
    for i in range(len(scores)):
        if scores[i] >= threhold and user_ids[i] in truth_ids: 
            count+=1
    return f"{count}/{k}"

def auc_score(scores, user_ids, truth_ids):
    threhold = heapq.nlargest(len(truth_ids), scores)[-1]
    predictions, labels = [0] * len(scores), [0] * len(scores)

    for i in range(len(scores)):
        if scores[i] >= threhold and user_ids[i] in truth_ids: predictions[i] = 1
    for id in truth_ids:
        try: 
            labels[user_ids.index(id)] = 1
        except: 
            pass
    return f"The AUC score is {roc_auc_score(labels, predictions)}"
