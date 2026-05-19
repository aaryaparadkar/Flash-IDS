from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
import logging
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "debug": "dim",
    "success": "bold green",
    "step": "bold magenta"
})

console = Console(theme=custom_theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("flash-cadets")
logger.setLevel(logging.INFO)

console.print("[bold cyan]Flash-IDS: Cadets Dataset Evaluation[/bold cyan]", style="success")
logger.info(f"Device: {device}")
logger.info("Initializing...")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import os
import torch.nn.functional as F
import orjson as json
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
warnings.filterwarnings('ignore')
from torch_geometric.loader import NeighborLoader
import multiprocessing


# ═══════════════════════════════════════════════════════════
# CONFIGURATION — Edit these knobs before running the pipeline
# ═══════════════════════════════════════════════════════════

# ── Mode flags ──
Train = True                         # True=train, False=only inference
USE_SYNTHETIC = False                # True=random data, False=real CDM JSONL

# ── Input JSONL paths (CDM format, one datum per line) ──
TRAIN_JSONL_PATHS = ['cadets_sampled/train.jsonl']   # List of training JSONL file paths
TEST_JSONL_PATHS = ['cadets_sampled/test.jsonl']    # List of test JSONL file paths (optional)

# ── Sampling (set to None for full file) ──
SAMPLE_LINES = None                  # Number of lines to sample from each file
SAMPLE_MODE = 'head'                 # 'head' (first N) or 'random'

# ── Train/test split (used only when TEST_JSONL_PATHS is empty) ──
TRAIN_RATIO = 0.8                    # Fraction of data for training

# ── Output paths ──
TRAIN_TSV = 'cadets_train.txt'
TEST_TSV = 'cadets_test.txt'
W2V_MODEL_PATH = 'trained_weights/cadets/word2vec_cadets_E3.model'
GNN_SNAPSHOT_PREFIX = 'trained_weights/cadets/lword2vec_gnn_cadets'
NUM_SNAPSHOTS = 22                   # Number of GNN ensemble snapshots

# ── Word2Vec hyperparams ──
W2V_VECTOR_SIZE = 30
W2V_WINDOW = 5
W2V_EPOCHS = 300

# ── GNN hyperparams ──
GNN_HIDDEN = 32
GNN_DROPOUT = 0.5
GNN_LR = 0.01
GNN_WEIGHT_DECAY = 5e-4
GNN_BATCH_SIZE = 5000

# ── Label map (CDM types -> class indices) ──
LABEL_MAP = {
    'SUBJECT_PROCESS': 0,
    'FILE_OBJECT_FILE': 1,
    'FILE_OBJECT_UNIX_SOCKET': 2,
    'UnnamedPipeObject': 3,
    'NetFlowObject': 4,
    'FILE_OBJECT_DIR': 5,
}

# ═══════════════════════════════════════════════════════════
# SAMPLING + VALIDATION UTILITIES
# ═══════════════════════════════════════════════════════════

def sample_jsonl_lines(input_path, output_path, num_lines, mode='head'):
    """Sample N lines from a JSONL file into output_path."""
    import random
    logger.info(f"Sampling {num_lines} lines from {input_path} ({mode} mode)")
    with open(input_path, 'r') as f:
        if mode == 'head':
            lines = []
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                lines.append(line)
        elif mode == 'random':
            all_lines = f.readlines()
            total = len(all_lines)
            k = min(num_lines, total)
            indices = set(random.sample(range(total), k))
            lines = [all_lines[i] for i in sorted(indices)]
        else:
            raise ValueError(f"Unknown sample mode: {mode}")
    logger.info(f"  Wrote {len(lines)} lines to {output_path}")
    with open(output_path, 'w') as f:
        f.writelines(lines)
    return len(lines)

def split_jsonl_by_time(input_path, train_path, test_path, ratio=0.8):
    """Split JSONL into train/test by line ratio (assumes time-ordering)."""
    logger.info(f"Splitting {input_path} into train ({ratio:.0%}) / test ({1-ratio:.0%})")
    with open(input_path, 'r') as f:
        lines = f.readlines()
    total = len(lines)
    split_idx = int(total * ratio)
    with open(train_path, 'w') as f:
        f.writelines(lines[:split_idx])
    with open(test_path, 'w') as f:
        f.writelines(lines[split_idx:])
    logger.info(f"  Train: {split_idx} lines, Test: {total - split_idx} lines")
    return split_idx, total - split_idx

def validate_cdm_jsonl(file_path):
    """Validate a JSONL file has CDM Event records and returns counts."""
    total = 0
    events = 0
    hosts = 0
    others = 0
    sample_line = None
    with open(file_path, 'r') as f:
        for line in f:
            total += 1
            if sample_line is None:
                sample_line = line[:200]
            if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                events += 1
            elif 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                hosts += 1
            else:
                others += 1
    logger.info(f"CDM validation for {file_path}:")
    logger.info(f"  Total lines: {total}, Events: {events}, Hosts: {hosts}, Other: {others}")
    if events == 0:
        logger.warning("  ⚠ No Event records found — cannot build edges!")
    return {'total': total, 'events': events, 'hosts': hosts, 'others': others}

def validate_edge_tsv(file_path):
    """Validate tab-separated edge file has correct schema."""
    import pandas as pd
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    df = pd.read_csv(file_path, sep='\t', header=None,
                     names=['actorID','actor_type','objectID','object_type','action','timestamp'])
    logger.info(f"Edge TSV validation for {file_path}:")
    logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    logger.info(f"  Node types: {df['actor_type'].unique().tolist()}")
    logger.info(f"  Action types: {df['action'].unique().tolist()}")
    return df

from pprint import pprint
import gzip
from sklearn.manifold import TSNE
import json
import copy
import os

import re

def extract_uuid(line):
    pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
    return pattern_uuid.findall(line)

def extract_subject_type(line):
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    return pattern_type.findall(line)

def show(file_path):
    logger.info(f"Processing: {file_path}")

def extract_edge_info(line):
    pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    pattern_time = re.compile(r'timestampNanos\":(.*?),')

    edge_type = extract_subject_type(line)[0]
    timestamp = pattern_time.findall(line)[0]
    src_id = pattern_src.findall(line)

    if len(src_id) == 0:
        return None, None, None, None, None

    src_id = src_id[0]
    dst_id1 = pattern_dst1.findall(line)
    dst_id2 = pattern_dst2.findall(line)

    if len(dst_id1) > 0 and dst_id1[0] != 'null':
        dst_id1 = dst_id1[0]
    else:
        dst_id1 = None

    if len(dst_id2) > 0 and dst_id2[0] != 'null':
        dst_id2 = dst_id2[0]
    else:
        dst_id2 = None

    return src_id, edge_type, timestamp, dst_id1, dst_id2

def process_data(file_path):
    id_nodetype_map = {}
    notice_num = 1000000
    logger.info(f"Starting data processing: {file_path}")
    for i in range(100):
        now_path = file_path + '.' + str(i)
        if i == 0:
            now_path = file_path
        if not os.path.exists(now_path):
            break

        with open(now_path, 'r') as f:
            show(now_path)
            cnt = 0
            total_lines = 0
            for line in f:
                cnt += 1
                total_lines += 1
                if cnt % notice_num == 0:
                    logger.debug(f"Processed {cnt:,} lines...")

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                    continue

                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                    continue

                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                    continue

                uuid = extract_uuid(line)[0]
                subject_type = extract_subject_type(line)

                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        id_nodetype_map[uuid] = 'MemoryObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        id_nodetype_map[uuid] = 'NetFlowObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        id_nodetype_map[uuid] = 'UnnamedPipeObject'
                        continue

                id_nodetype_map[uuid] = subject_type[0]

    logger.info(f"Data processing complete: {len(id_nodetype_map)} nodes extracted")
    return id_nodetype_map

def process_edges(file_path, id_nodetype_map):
    notice_num = 1000000
    not_in_cnt = 0
    logger.info(f"Starting edge processing: {file_path}")

    for i in range(100):
        now_path = file_path + '.' + str(i)
        if i == 0:
            now_path = file_path
        if not os.path.exists(now_path):
            break

        with open(now_path, 'r') as f, open(now_path+'.txt', 'w') as fw:
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    logger.debug(f"Processed {cnt:,} edges...")

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    src_id, edge_type, timestamp, dst_id1, dst_id2 = extract_edge_info(line)

                    if src_id is None or src_id not in id_nodetype_map:
                        not_in_cnt += 1
                        continue

                    src_type = id_nodetype_map[src_id]

                    if dst_id1 is not None and dst_id1 in id_nodetype_map:
                        dst_type1 = id_nodetype_map[dst_id1]
                        this_edge1 = f"{src_id}\t{src_type}\t{dst_id1}\t{dst_type1}\t{edge_type}\t{timestamp}\n"
                        fw.write(this_edge1)

                    if dst_id2 is not None and dst_id2 in id_nodetype_map:
                        dst_type2 = id_nodetype_map[dst_id2]
                        this_edge2 = f"{src_id}\t{src_type}\t{dst_id2}\t{dst_type2}\t{edge_type}\t{timestamp}\n"
                        fw.write(this_edge2)

def run_data_processing():
    if USE_SYNTHETIC:
        create_synthetic_data()
        return
    
    logger.info("Checking for dataset files...")
    
    tar_file_1 = 'ta1-cadets-e3-official.json.tar.gz'
    tar_file_2 = 'ta1-cadets-e3-official-2.json.tar.gz'
    json_file_1 = 'ta1-cadets-e3-official.json'
    json_file_2 = 'ta1-cadets-e3-official-2.json'
    
    # Check if already processed
    if os.path.exists('cadets_train.txt') and os.path.exists('cadets_test.txt'):
        logger.info("Processed data already exists! Skipping data processing.")
        return
    
    # Check for raw JSON files (already extracted)
    if os.path.exists(json_file_1) and os.path.exists(json_file_2):
        logger.info("Found extracted JSON files, processing...")
    elif not os.path.exists(tar_file_1) or not os.path.exists(tar_file_2):
        logger.error(f"Missing dataset files! Please download them first.")
        logger.info(f"Expected files: {tar_file_1}, {tar_file_2}")
        logger.info("Or download from: https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK")
        return
    else:
        logger.info("Extracting dataset archives...")
        os.system(f'tar -zxvf {tar_file_1}')
        os.system(f'tar -zxvf {tar_file_2}')

    path_list = [json_file_1, json_file_2]

    for path in path_list:
        if os.path.exists(path):
            id_nodetype_map = process_data(path)
            process_edges(path, id_nodetype_map)
        else:
            logger.warning(f"File not found: {path}")

    if os.path.exists('ta1-cadets-e3-official.json.1.txt'):
        os.system('cp ta1-cadets-e3-official.json.1.txt cadets_train.txt')
        logger.info("Created cadets_train.txt")
    else:
        logger.warning("Could not create cadets_train.txt")
    
    if os.path.exists('ta1-cadets-e3-official-2.json.txt'):
        os.system('cp ta1-cadets-e3-official-2.json.txt cadets_test.txt')
        logger.info("Created cadets_test.txt")
    else:
        logger.warning("Could not create cadets_test.txt")

def create_synthetic_data():
    """Create synthetic test data for smoke testing without downloading datasets."""
    import random
    import uuid
    
    logger.info("Creating synthetic test data (smoke test mode)...")
    
    # Generate synthetic train/test data
    # Format: actorID, actor_type, objectID, object_type, action, timestamp
    node_types = ['SUBJECT_PROCESS', 'FILE_OBJECT_FILE', 'FILE_OBJECT_UNIX_SOCKET', 
               'UnnamedPipeObject', 'NetFlowObject', 'FILE_OBJECT_DIR']
    actions = ['exec', 'open', 'read', 'write', 'fork', 'network']
    
    def generate_synthetic_data(num_lines, is_test=False):
        lines = []
        for i in range(num_lines):
            actor = f"proc_{random.randint(1, 100)}"
            obj = f"file_{random.randint(1, 100)}" if random.random() > 0.5 else f"sock_{random.randint(1, 50)}"
            actor_type = node_types[random.randint(0, 2)]
            obj_type = node_types[random.randint(3, 5)]
            action = actions[random.randint(0, len(actions)-1)]
            timestamp = str(1000000 + i)
            lines.append(f"{actor}\t{actor_type}\t{obj}\t{obj_type}\t{action}\t{timestamp}\n")
        return lines
    
    # Create train data
    train_lines = generate_synthetic_data(1000)
    with open('cadets_train.txt', 'w') as f:
        f.writelines(train_lines)
    logger.info(f"Created cadets_train.txt with {len(train_lines)} lines")
    
    # Create test data
    test_lines = generate_synthetic_data(500, is_test=True)
    with open('cadets_test.txt', 'w') as f:
        f.writelines(test_lines)
    logger.info(f"Created cadets_test.txt with {len(test_lines)} lines")
    
    # Also create ground truth for testing
    import json
    malicious = [f"proc_{i}" for i in range(1, 11)]  # First 10 processes as malicious
    with open('data_files/cadets.json', 'w') as f:
        json.dump(malicious, f)
    logger.info("Created synthetic ground truth")
    logger.info("Smoke test data ready!")

def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)

def update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

def prepare_graph(df):
    nodes, labels, edges = {}, {}, []
    dummies = LABEL_MAP
    
    for _, row in df.iterrows():
        action = row["action"]
        properties = [row['exec'], action] + ([row['path']] if row['path'] else [])
        
        actor_id = row["actorID"]
        add_node_properties(nodes, actor_id, properties)
        labels[actor_id] = dummies[row['actor_type']]

        object_id = row["objectID"]
        add_node_properties(nodes, object_id, properties)
        labels[object_id] = dummies[row['object']]

        edges.append((actor_id, object_id))

    features, feat_labels, edge_index, index_map = [], [], [[], []], {}
    for node_id, props in nodes.items():
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1

    update_edge_index(edges, edge_index, index_map)

    return features, feat_labels, edge_index, list(index_map.keys())

from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, in_channel, out_channel, hidden=GNN_HIDDEN, dropout=GNN_DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channel, hidden, normalize=True)
        self.conv2 = SAGEConv(hidden, out_channel, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

from gensim.models.callbacks import CallbackAny2Vec
import gensim
from gensim.models import Word2Vec
from multiprocessing import Pool
from itertools import compress
from tqdm import tqdm
import time

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        model.save(W2V_MODEL_PATH)
        self.epoch += 1

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        logger.debug(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        logger.debug(f"Epoch #{self.epoch} end")
        self.epoch += 1

epoch_logger = EpochLogger()
saver = EpochSaver()

def add_attributes(d, p):
    """Enrich edge DataFrame with exec/path attributes from raw CDM JSONL.
    For synthetic mode or missing file, returns data as-is."""
    if USE_SYNTHETIC:
        logger.debug("Synthetic mode: skipping attribute enrichment")
        return d
    if not os.path.exists(str(p)):
        logger.warning(f"Raw JSON not found at {p}, adding default attributes")
        d['exec'] = ''
        d['path'] = ''
        return d

    EVENT_KEY = 'com.bbn.tc.schema.avro.cdm18.Event'
    info = []
    with open(p, 'r') as f:
        for line in f:
            if EVENT_KEY not in line:
                continue
            try:
                x = json.loads(line)
                ev = x['datum'][EVENT_KEY]
                action = ev.get('type', '')
                actor = ev.get('subject', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')
                obj = ev.get('predicateObject', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')
                timestamp = str(ev.get('timestampNanos', ''))
                cmd = ev.get('properties', {}).get('map', {}).get('exec', '')
                path = ev.get('predicateObjectPath', {}).get('string', '')
                path2 = ev.get('predicateObject2Path', {}).get('string', '')
                obj2 = ev.get('predicateObject2', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')
                if obj2:
                    info.append({'actorID':actor,'objectID':obj2,'action':action,
                                 'timestamp':timestamp,'exec':cmd, 'path':path2})
                info.append({'actorID':actor,'objectID':obj,'action':action,
                             'timestamp':timestamp,'exec':cmd, 'path':path})
            except Exception:
                continue

    if not info:
        logger.warning("No events enriched from raw CDM; adding default columns")
        d['exec'] = ''
        d['path'] = ''
        return d

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)
    result = d.merge(rdf, how='left', on=['actorID','objectID','action','timestamp'])
    logger.info(f"Attribute enrichment: {len(d)} edges -> {len(result)} enriched ({result['exec'].notna().sum()} with exec)")
    return result

import math
import torch
import numpy as np
from gensim.models import Word2Vec

class PositionalEncoder:

    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]


def infer(document):
    """Infer embeddings for a document. For synthetic mode, returns random embeddings."""
    if USE_SYNTHETIC or w2vmodel is None:
        logger.debug("Synthetic mode: using random embeddings")
        return np.random.randn(W2V_VECTOR_SIZE).astype(np.float32)
    
    word_embeddings = [w2vmodel.wv[word] for word in document if word in  w2vmodel.wv]
    
    if not word_embeddings:
        return np.zeros(W2V_VECTOR_SIZE)

    output_embedding = torch.tensor(word_embeddings, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)

encoder = PositionalEncoder(W2V_VECTOR_SIZE)

# Load Word2Vec model (or None for synthetic / first-run mode)
if USE_SYNTHETIC or not os.path.exists(W2V_MODEL_PATH):
    w2vmodel = None
    if not USE_SYNTHETIC and not os.path.exists(W2V_MODEL_PATH):
        logger.info("No Word2Vec model found yet — will train from scratch")
    elif USE_SYNTHETIC:
        logger.info("Synthetic mode: using random embeddings instead of Word2Vec")
else:
    w2vmodel = Word2Vec.load(W2V_MODEL_PATH)

from itertools import compress
from torch_geometric import utils

def Get_Adjacent(ids, mapp, edges, hops):
    if hops == 0:
        return set()
    
    neighbors = set()
    for edge in zip(edges[0], edges[1]):
        if any(mapp[node] in ids for node in edge):
            neighbors.update(mapp[node] for node in edge)

    if hops > 1:
        neighbors = neighbors.union(Get_Adjacent(neighbors, mapp, edges, hops - 1))
    
    return neighbors

def calculate_metrics(TP, FP, FN, TN):
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    TPR = TP / (TP + FN) if TP + FN > 0 else 0

    prec = TP / (TP + FP) if TP + FP > 0 else 0
    rec = TP / (TP + FN) if TP + FN > 0 else 0
    fscore = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0

    return prec, rec, fscore, FPR, TPR

def helper(MP, all_pids, GP, edges, mapp):
    logger.info("Calculating evaluation metrics...")
    TP = MP.intersection(GP)
    FP = MP - GP
    FN = GP - MP
    TN = all_pids - (GP | MP)

    two_hop_gp = Get_Adjacent(GP, mapp, edges, 2)
    two_hop_tp = Get_Adjacent(TP, mapp, edges, 2)
    FPL = FP - two_hop_gp
    TPL = TP.union(FN.intersection(two_hop_tp))
    FN = FN - two_hop_tp

    TP, FP, FN, TN = len(TPL), len(FPL), len(FN), len(TN)

    prec, rec, fscore, FPR, TPR = calculate_metrics(TP, FP, FN, TN)
    logger.info(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
    console.print(f"[bold green]Precision:[/bold green] {round(prec, 2)}, [bold green]Recall:[/bold green] {round(rec, 2)}, [bold green]Fscore:[/bold green] {round(fscore, 2)}")
    
    return TPL, FPL

# ═══════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR — one-call training entry point
# ═══════════════════════════════════════════════════════════

def run_training_pipeline(
    train_paths=None,
    test_paths=None,
    sample_lines=None,
    sample_mode='head',
    train_ratio=0.8,
    train_tsv='cadets_train.txt',
    test_tsv='cadets_test.txt',
    w2v_path='trained_weights/cadets/word2vec_cadets_E3.model',
    num_snapshots=22,
    embed_mode='baseline',
    hf_model='BAAI/bge-small-en-v1.5',
    hf_batch_size=64,
    hf_max_length=128,
):
    """
    End-to-end CDM JSONL training pipeline.
    
    Accepts one or more raw CDM JSONL files and runs:
      1. Optional sampling
      2. Input validation
      3. Train/test split (if only train_paths given)
      4. CDM -> edge TSV parsing
      5. Attribute enrichment
      6. Word2Vec training
      7. Multi-snapshot GNN training
      8. Inference + evaluation on test set
    """
    logger.info("═" * 60)
    logger.info("FLASH Training Pipeline — Starting")
    logger.info("═" * 60)

    # ── Resolve paths ──
    if train_paths is None:
        train_paths = TRAIN_JSONL_PATHS
    if test_paths is None:
        test_paths = TEST_JSONL_PATHS
    if not isinstance(train_paths, list):
        train_paths = [train_paths]
    if isinstance(test_paths, str):
        test_paths = [test_paths]

    logger.info(f"Train files: {train_paths}")
    logger.info(f"Test files: {test_paths or 'None (will split)'}")

    # ── Step 1: Validate inputs ──
    for p in train_paths:
        if not os.path.exists(p):
            logger.error(f"Train file not found: {p}")
            return
        validate_cdm_jsonl(p)
    for p in (test_paths or []):
        if not os.path.exists(p):
            logger.error(f"Test file not found: {p}")
            return
        validate_cdm_jsonl(p)

    # ── Step 2: Sample or concat train files ──
    if len(train_paths) == 1 and sample_lines:
        sampled = f"{train_paths[0]}.sampled"
        sample_jsonl_lines(train_paths[0], sampled, sample_lines, sample_mode)
        train_paths = [sampled]
    elif len(train_paths) > 1:
        combined = 'combined_train.jsonl'
        with open(combined, 'w') as out:
            for p in train_paths:
                with open(p) as f:
                    for line in f:
                        out.write(line)
        logger.info(f"Combined {len(train_paths)} files into {combined}")
        if sample_lines:
            sampled = 'combined_train_sampled.jsonl'
            sample_jsonl_lines(combined, sampled, sample_lines, sample_mode)
            os.remove(combined)
            train_paths = [sampled]
        else:
            train_paths = [combined]

    # ── Step 3: Split if no test files ──
    if not test_paths:
        raw_train = f"{train_paths[0]}.raw_train"
        raw_test = f"{train_paths[0]}.raw_test"
        split_jsonl_by_time(train_paths[0], raw_train, raw_test, train_ratio)
        train_paths = [raw_train]
        test_paths = [raw_test]

    # ── Step 4: Parse CDM -> edge TSV ──
    logger.info("")
    logger.info("─" * 40)
    logger.info("Step 4: Parsing CDM to edge TSV")
    logger.info("─" * 40)
    
    # Process train
    for p in train_paths:
        id_map = process_data(p)
        process_edges(p, id_map)
    os.system(f'cp {train_paths[0]}.txt {train_tsv}')
    logger.info(f"Created {train_tsv}")
    train_df = validate_edge_tsv(train_tsv)
    if train_df is None or len(train_df) == 0:
        logger.error("Train TSV is empty — cannot train")
        return

    # Process test
    for p in test_paths:
        id_map = process_data(p)
        process_edges(p, id_map)
    os.system(f'cp {test_paths[0]}.txt {test_tsv}')
    logger.info(f"Created {test_tsv}")
    test_df = validate_edge_tsv(test_tsv)
    if test_df is None or len(test_df) == 0:
        logger.error("Test TSV is empty")
        return

    # ── Step 5: Attribute enrichment + Graph build ──
    logger.info("")
    logger.info("─" * 40)
    logger.info("Step 5: Building training graph")
    logger.info("─" * 40)
    
    raw_train_path = train_paths[0]
    df = pd.read_csv(train_tsv, sep='\t', header=None,
                     names=['actorID','actor_type','objectID','object','action','timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = add_attributes(df, raw_train_path)
    phrases, labels, edges, mapp = prepare_graph(df)
    logger.info(f"Graph: {len(phrases)} nodes, {len(edges)} edges, {len(set(labels))} classes")

    # ── Step 6: Embedding preparation + Word2Vec training ──
    logger.info("")
    logger.info("─" * 40)
    logger.info(f"Step 6: Preparing embeddings (mode={embed_mode})")
    logger.info("─" * 40)

    from flash_embed import get_provider as _get_provider, batch_event_to_text

    if embed_mode == 'hf':
        _hf_provider = _get_provider('hf', model_name=hf_model,
                                      batch_size=hf_batch_size,
                                      max_length=hf_max_length,
                                      cache_dir='.hf_cache')
        _emb_dim = _hf_provider.vector_size
        logger.info(f"Using HF provider: {hf_model} (dim={_emb_dim})")
        logger.info(f"  Embedding {len(phrases)} training phrases...")
        _train_texts = batch_event_to_text(phrases)
        nodes = _hf_provider.embed_batch(_train_texts)
        logger.info(f"  Done — shape={nodes.shape}")
    else:
        _emb_dim = W2V_VECTOR_SIZE
        if Train and not USE_SYNTHETIC:
            logger.info("Step 6a: Training Word2Vec")
            word2vec = Word2Vec(sentences=phrases, vector_size=W2V_VECTOR_SIZE,
                               window=W2V_WINDOW, min_count=1, workers=8,
                               epochs=W2V_EPOCHS, callbacks=[saver, epoch_logger])
            word2vec.save(w2v_path)
            logger.info(f"Word2Vec model saved to {w2v_path}")
        nodes = [infer(x) for x in phrases]
        nodes = np.array(nodes)

    # ── Step 7: GNN training ──
    if Train:
        logger.info("")
        logger.info("─" * 40)
        logger.info("Step 7: Training GNN snapshots")
        logger.info("─" * 40)
        
        from sklearn.utils import class_weight
        from torch.nn import CrossEntropyLoss

        model = GCN(_emb_dim, len(LABEL_MAP)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WEIGHT_DECAY)

        l = np.array(labels)
        class_weights = class_weight.compute_class_weight(class_weight=None, classes=np.unique(l), y=l)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')

        graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device),
                     y=torch.tensor(labels, dtype=torch.long).to(device),
                     edge_index=torch.tensor(edges, dtype=torch.long).to(device))
        graph.n_id = torch.arange(graph.num_nodes)
        mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool, device=device)

        for m_n in range(num_snapshots):
            loader = NeighborLoader(graph, num_neighbors=[-1, -1],
                                    batch_size=GNN_BATCH_SIZE, input_nodes=mask)
            total_loss = 0
            for subg in loader:
                model.train()
                optimizer.zero_grad()
                out = model(subg.x, subg.edge_index)
                loss = criterion(out, subg.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * subg.batch_size
            logger.info(f"Snapshot {m_n}/{num_snapshots-1} - Loss: {total_loss / mask.sum().item():.4f}")

            # Confidence-based masking
            loader = NeighborLoader(graph, num_neighbors=[-1, -1],
                                    batch_size=GNN_BATCH_SIZE, input_nodes=mask)
            for subg in loader:
                model.eval()
                out = model(subg.x, subg.edge_index)
                sorted, indices = out.sort(dim=1, descending=True)
                conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
                conf = (conf - conf.min()) / conf.max()
                pred = indices[:, 0]
                cond = (pred == subg.y) | (conf >= 0.9)
                mask[subg.n_id[cond.cpu()].to(mask.device)] = False

            snap_path = f'{GNN_SNAPSHOT_PREFIX}{m_n}_E3.pth'
            torch.save(model.state_dict(), snap_path)
            remaining = mask.sum().item()
            logger.info(f"  Snapshot saved to {snap_path} — {remaining} nodes remain")

    # ── Step 8: Evaluation ──
    logger.info("")
    logger.info("─" * 40)
    logger.info("Step 8: Evaluating on test set")
    logger.info("─" * 40)
    
    raw_test_path = test_paths[0]
    test_df = pd.read_csv(test_tsv, sep='\t', header=None,
                          names=['actorID','actor_type','objectID','object','action','timestamp'])
    test_df = test_df.dropna()
    test_df.sort_values(by='timestamp', ascending=True, inplace=True)
    test_df = add_attributes(test_df, raw_test_path)
    eval_phrases, eval_labels, eval_edges, eval_mapp = prepare_graph(test_df)
    logger.info(f"Test graph: {len(eval_phrases)} nodes, {len(eval_edges)} edges")

    if embed_mode == 'hf':
        logger.info(f"  Embedding {len(eval_phrases)} test phrases via HF...")
        _eval_texts = batch_event_to_text(eval_phrases)
        eval_nodes = _hf_provider.embed_batch(_eval_texts)
    else:
        eval_nodes = [infer(x) for x in eval_phrases]
        eval_nodes = np.array(eval_nodes)

    eval_graph = Data(x=torch.tensor(eval_nodes, dtype=torch.float).to(device),
                      y=torch.tensor(eval_labels, dtype=torch.long).to(device),
                      edge_index=torch.tensor(eval_edges, dtype=torch.long).to(device))
    eval_graph.n_id = torch.arange(eval_graph.num_nodes, device=device)
    flag = torch.tensor([True] * eval_graph.num_nodes, dtype=torch.bool, device=device)

    for m_n in range(num_snapshots):
        snap_path = f'{GNN_SNAPSHOT_PREFIX}{m_n}_E3.pth'
        if not os.path.exists(snap_path):
            logger.warning(f"Snapshot {snap_path} not found, skipping")
            continue
        model.load_state_dict(torch.load(snap_path))
        loader = NeighborLoader(eval_graph, num_neighbors=[-1, -1], batch_size=GNN_BATCH_SIZE)
        for subg in loader:
            model.eval()
            out = model(subg.x, subg.edge_index)
            sorted, indices = out.sort(dim=1, descending=True)
            conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
            conf = (conf - conf.min()) / conf.max()
            pred = indices[:, 0]
            cond = (pred == subg.y)
            nid_cond = subg.n_id[cond]
            flag[nid_cond] = torch.zeros_like(flag[nid_cond], dtype=torch.bool)

    from itertools import compress
    index = utils.mask_to_index(flag).tolist()
    ids = set([eval_mapp[x] for x in index])
    
    # Load ground truth if available
    gt_path = 'data_files/cadets.json'
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as jf:
            GT_mal = set(json.load(jf))
    else:
        GT_mal = set()
        logger.warning("No ground truth file found, using empty GT")

    all_ids = list(test_df['actorID']) + list(test_df['objectID'])
    all_ids = set(all_ids)
    TPL, FPL = helper(set(ids), set(all_ids), GT_mal, eval_edges, eval_mapp)

    logger.info("═" * 60)
    logger.info("Pipeline complete!")
    logger.info("═" * 60)
    return {
        'train_nodes': len(phrases),
        'train_edges': len(edges),
        'test_nodes': len(eval_phrases),
        'test_edges': len(eval_edges),
        'snapshots': num_snapshots,
    }

def traverse(ids, mapping, edges, hops, visited=None):
    if hops == 0:
        return set()

    if visited is None:
        visited = set()

    neighbors = set()
    for src, dst in zip(edges[0], edges[1]):
        src_mapped, dst_mapped = mapping[src], mapping[dst]

        if (src_mapped in ids and dst_mapped not in visited) or \
           (dst_mapped in ids and src_mapped not in visited):
            neighbors.add(src_mapped)
            neighbors.add(dst_mapped)

        visited.add(src_mapped)
        visited.add(dst_mapped)

    neighbors.difference_update(ids) 
    return ids.union(traverse(neighbors, mapping, edges, hops - 1, visited))

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_connected_alerts(start_alert, mapping, edges, depth, remaining_alerts):
    connected_path = traverse({start_alert}, mapping, edges, depth)
    return connected_path.intersection(remaining_alerts)

def generate_incident_graphs(alerts, edges, mapping, depth):
    incident_graphs = []
    remaining_alerts = set(alerts)

    while remaining_alerts:
        alert = remaining_alerts.pop()
        connected_alerts = find_connected_alerts(alert, mapping, edges, depth, remaining_alerts)

        if len(connected_alerts) > 1:
            incident_graphs.append(connected_alerts)
            remaining_alerts -= connected_alerts

    return incident_graphs



# ═══════════════════════════════════════════════════════════
# EXECUTE: run the full pipeline with current config
# ═══════════════════════════════════════════════════════════

import time as _t

def _ensure_exec_path_columns(df):
    """Add exec/path with safe defaults if missing from edge DataFrame."""
    if 'exec' not in df.columns:
        df['exec'] = ''
    if 'path' not in df.columns:
        df['path'] = ''
    return df

# Patch prepare_graph to be safe
_original_prepare = prepare_graph
def _safe_prepare_graph(df):
    df = _ensure_exec_path_columns(df)
    return _original_prepare(df)
prepare_graph = _safe_prepare_graph

if __name__ == '__main__':
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(description="Flash-IDS Training Pipeline")
    _parser.add_argument("--embed-mode", default="baseline",
                         choices=["baseline", "hf"],
                         help="Embedding backend: baseline (Word2Vec) or hf (Hugging Face)")
    _parser.add_argument("--hf-model", default="BAAI/bge-small-en-v1.5",
                         help="Hugging Face model ID for --embed-mode hf")
    _parser.add_argument("--hf-batch-size", type=int, default=64,
                         help="Batch size for HF API calls")
    _parser.add_argument("--hf-max-length", type=int, default=128,
                         help="Max token length for HF model")
    _args = _parser.parse_args()

    if _args.embed_mode == "hf":
        if not os.environ.get("HF_TOKEN"):
            console.print("[bold red]Error: HF_TOKEN environment variable not set.[/bold red]")
            console.print("Get a token at https://huggingface.co/settings/tokens")
            console.print("Set it with: export HF_TOKEN=hf_...")
            sys.exit(1)

    _start = _t.time()
    result = run_training_pipeline(
        embed_mode=_args.embed_mode,
        hf_model=_args.hf_model,
        hf_batch_size=_args.hf_batch_size,
        hf_max_length=_args.hf_max_length,
    )
    _elapsed = _t.time() - _start
    
    if result:
        result['pipeline_elapsed_seconds'] = round(_elapsed, 2)
        result['config'] = {
            'train_paths': TRAIN_JSONL_PATHS,
            'test_paths': TEST_JSONL_PATHS,
            'w2v_vector_size': W2V_VECTOR_SIZE,
            'w2v_window': W2V_WINDOW,
            'w2v_epochs': W2V_EPOCHS,
            'gnn_hidden': GNN_HIDDEN,
            'gnn_dropout': GNN_DROPOUT,
            'gnn_lr': GNN_LR,
            'gnn_weight_decay': GNN_WEIGHT_DECAY,
            'gnn_batch_size': GNN_BATCH_SIZE,
            'num_snapshots': NUM_SNAPSHOTS,
            'embed_mode': _args.embed_mode,
            'hf_model': _args.hf_model if _args.embed_mode == 'hf' else None,
        }
        
        import json as _json
        _bf = f'benchmark_results_{_t.strftime("%Y%m%d_%H%M%S")}.json'
        with open(_bf, 'w') as _f:
            _json.dump(result, _f, indent=2)
        print(f"\nBenchmark results saved to {_bf}")
        print(_json.dumps(result, indent=2))
