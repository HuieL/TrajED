from tqdm import  tqdm
import pickle as pkl
import warnings
from transformers import BertTokenizer, BertModel

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


def text_trajectory(df, userid):
    weekdayDict = {
        0 : 'Monday', 1 : 'Tuesday', 2 : 'Wednesday', 3 : 'Thursday', 4 : 'Friday', 5 : 'Saturday', 6 : 'Sunday',
    }
    item = df.loc[df['AgentID']==userid]
    item['ArrivingTime'] = item['ArrivingTime'].str.replace(',', ' ')
    item['ArrivingTime'] = pd.to_datetime(item['ArrivingTime'])
    item['dayofweek'] = item.ArrivingTime.apply(lambda x: weekdayDict[x.dayofweek])
    st_sequence, text_sequence, sequence = [], [], ""
    prev_X, prev_Y = None, None
    index = [i for i in range(len(item))]
    for i in range(len(item)):
        sequence = ""
        cur_item = item.iloc[i]
        CheckinTime, VenueType, dayofweek, X, Y = cur_item['ArrivingTime'], cur_item['LocationType'], cur_item['dayofweek'], cur_item['Longitude'], cur_item['Latitude']

        # form trajectory sequnece
        time = ':'.join(str(CheckinTime).split(' ')[1].split(':'))
        sequence += f"{dayofweek} {str(time)}, {VenueType}"

        st_sequence.append([np.asarray([X, Y]), time])
        text_sequence.append(sequence)

    return st_sequence, text_sequence, index

path = ".\dataset\geolife\geolife-dataset_full-20-agents-0.8-normal-portion.tsv"
df = pd.read_csv(path, sep = " ")
n_neighbors = 1

sampling_ids = torch.load(".\dataset\geolife\sampled_ids.pt")
user_id, st_sequence, text_sequence, index, label = [], [], [], [], []
for j, uid in enumerate(tqdm(sampling_ids)):
    st, text, id = text_trajectory(df, uid)
    converted_text =  torch.stack([bert_embeddings(text[i]) for i in range(len(text))])
    st_sequence.append(st), text_sequence.append(converted_text), index.append(id), user_id.append(int(uid))
    if uid in all_ids: label.append('abnormal')
    else: label.append('normal')

users = [{'id': user_id[i], 'st_sequence': st_sequence[i], 'text_sequence': text_sequence[i], 'length': len(index[i]), 'label': label[i]} for i in range(len(label))]
with open(r'C:\Users\HuieL\VScodes\TrajectoryDistiallation\datasets\geolife\data_info.pkl', 'wb') as handle:
    pkl.dump(users, handle)
