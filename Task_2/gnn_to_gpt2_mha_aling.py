# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
import networkx as nx
import penman
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer,GPT2Config
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.optim as optim

DEVICE = torch.device("cuda:0")


class TextGraphDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, graphs, amr_ids,amr_mask):
        assert len(input_ids) == len(graphs), "Text and graph data must have the same length"
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.graphs = graphs
        self.amr_ids=amr_ids
        self.amr_mask=amr_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "graph": self.graphs[idx],
            "amr_ids":self.amr_ids[idx],
            "amr_mask":self.amr_mask[idx],
        }


def make_amr_graph_lists(data_file):
    amr_graphs = []
    text_list = []
    amr_string_list=[]
    with open(data_file, 'r', encoding='utf-8') as f:
        amr_data_train = f.read()
    graphs = list(penman.loads(amr_data_train))

    for i, g in enumerate(graphs):
        nx_graph = nx.DiGraph()
        text_list.append(g.metadata.get('snt', None))
        amr_string=penman.encode(g)
        amr_string_list.append(amr_string)
        for source, relation, target in g.triples:
            nx_graph.add_edge(source, target, label=relation)
        amr_graphs.append(nx_graph)
    return amr_graphs, text_list, amr_string_list


amr_graphs, text_list,amr_string_list= make_amr_graph_lists('qald-9-amr-train.txt')
model_path = 'gpt2'
config = GPT2Config.from_pretrained("gpt2", add_cross_attention=True)
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2",config=config)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
bert_tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")


def tokenize_text(text_list, tokenizer):
    tokenized_defn = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokenized_defn['input_ids'].to(DEVICE)
    attn_mask = tokenized_defn['attention_mask'].to(DEVICE)
    return input_ids, attn_mask


def custom_collate_fn_with_mask(data_list):
    """
    Collate function with padding and a padding mask.
    """
    # Extract node features (x) and their sizes
    node_features = [data.x for data in data_list]
    batch_sizes = [data.x.size(0) for data in data_list]

    # Pad node features to the same number of nodes
    padded_x = pad_sequence(node_features, batch_first=True)  # Shape: (batch_size, max_num_nodes, node_feature_dim)

    # Create a padding mask
    max_num_nodes = padded_x.size(1)
    mask = torch.zeros(len(data_list), max_num_nodes, dtype=torch.bool)
    for i, size in enumerate(batch_sizes):
        mask[i, :size] = 1  # Mark valid nodes as True

    return padded_x, mask




def custom_collate_fn(batch):
    # Extract text and AMR data
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    amr_ids = torch.stack([item["amr_ids"] for item in batch])
    amr_masks = torch.stack([item["amr_mask"] for item in batch])
    
    # Extract graph data
    graph_data_list = [item["graph"] for item in batch]
    node_features = [data.x for data in graph_data_list]
    edge_indices = [data.edge_index for data in graph_data_list]
    edge_types = [data.edge_type for data in graph_data_list]
    batch_sizes = [data.num_nodes for data in graph_data_list]
    
    # Pad node features
    padded_x = pad_sequence(node_features, batch_first=True)  # (batch_size, max_num_nodes, node_feature_dim)
    max_num_nodes = padded_x.size(1)

    # Pad edge indices and edge types
    padded_edge_index = []
    padded_edge_type = []
    offset = 0
    for edge_index, edge_type, size in zip(edge_indices, edge_types, batch_sizes):
        padded_edge_index.append(edge_index + offset)  # Offset to account for graph indices
        padded_edge_type.append(edge_type)
        offset += max_num_nodes

    padded_edge_index = torch.cat(padded_edge_index, dim=1)
    padded_edge_type = torch.cat(padded_edge_type, dim=0)

    # Create a padding mask for valid nodes
    mask = torch.zeros(len(graph_data_list), max_num_nodes, dtype=torch.bool)
    for i, size in enumerate(batch_sizes):
        mask[i, :size] = 1

    # Batch information for graphs
    batch_idx = torch.cat([torch.full((size,), i, dtype=torch.long) for i, size in enumerate(batch_sizes)])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "amr_ids": amr_ids,
        "amr_mask": amr_masks,
        "graphs": {
            "x": padded_x,
            "edge_index": padded_edge_index,
            "edge_type": padded_edge_type,
            "batch": batch_idx,
            "mask": mask,
        }
    }



def convert_amr_graph_to_data(graph, edge_label_map):
    # Create a local node-to-index mapping specific to this graph
    local_node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

    # Edge index and edge types
    edge_index = []
    edge_type = []
    for source, target, attr in graph.edges(data=True):
        # Map nodes to indices
        if source in local_node_to_idx and target in local_node_to_idx:
            edge_index.append((local_node_to_idx[source], local_node_to_idx[target]))
            label = attr.get("label", "relation")  # Default edge label if missing
            edge_type.append(edge_label_map[label])

    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # Node indices (used for embedding)
    node_indices = torch.arange(len(local_node_to_idx), dtype=torch.long)

    # Ensure all edge indices are within range
    max_index = node_indices.max().item() + 1
    if edge_index.max() >= max_index:
        raise ValueError(f"Found indices in 'edge_index' that exceed the range of valid nodes (0-{max_index - 1}).")

    return Data(x=node_indices, edge_index=edge_index, edge_type=edge_type)

def preprocess_graph_data(graphs, edge_label_map):
    """
    Preprocess graph data to create padded node features and padding masks.
    """
    processed_data = []
    max_nodes = max(len(graph.nodes) for graph in graphs)  # Find max number of nodes across all graphs

    for graph in graphs:
        # Convert graph to Data object
        data = convert_amr_graph_to_data(graph, edge_label_map)

        # Create a padding mask (1 for valid nodes, 0 for padding)
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        mask[:data.x.size(0)] = 1

        # Pad node features to max_nodes
        padded_x = torch.zeros(max_nodes, dtype=data.x.dtype)
        padded_x[:data.x.size(0)] = data.x

        # Update the graph Data object with padded features and mask
        data.x = padded_x
        data.mask = mask
        processed_data.append(data)

    return processed_data


# Preprocess graph data




unique_edge_labels = set(edge.get("label", "relation") for g in amr_graphs for _, _, edge in g.edges(data=True))
edge_label_map = {label: idx for idx, label in enumerate(unique_edge_labels)}
unique_nodes = {node for g in amr_graphs for node in g.nodes()}
node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

# data_list = [convert_amr_graph_to_data(g,edge_label_map) for g in amr_graphs]
data_list = preprocess_graph_data(amr_graphs, edge_label_map)

gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
input_ids, attn_mask = tokenize_text(text_list, gpt2_tokenizer)
amr_input_ids, amr_attn_mask=tokenize_text(amr_string_list,bert_tokenizer)
dataset = TextGraphDataset(input_ids, attn_mask, data_list, amr_input_ids,amr_attn_mask)
loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn, shuffle=True)


# GNN Encoder with nn.Embedding for nodes
class GNNEncoder(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, output_dim, num_relations):
        super(GNNEncoder, self).__init__()
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = RGCNConv(embedding_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations=num_relations)
        self.output_dim = output_dim
        self.global_pool = global_mean_pool

    def forward(self, node_indices, edge_index, edge_type,batch,mask):
        x = self.node_embedding(node_indices)
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        # x = self.global_pool(x, batch)
        x = x * mask.unsqueeze(-1)
        batch_size = batch.max().item() + 1  # Get the batch size
        num_nodes_per_graph = x.size(0) // batch_size  # Compute nodes per graph
        

# Reshape to (batch_size, num_nodes, hidden_channels)
        x_reshaped = x.view(batch_size, num_nodes_per_graph, -1)
        # print("x size in gcn",x.size())
        return x_reshaped # Aggregate node embeddings

class LinearizedGraphEnc(nn.Module):
    def __init__(self,output_dim):
        super(LinearizedGraphEnc, self).__init__()
        self.bert_model=BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert_model.parameters():
            p.requires_grad=False
        list(self.bert_model.parameters())[-1].requires_grad=True
        self.output_dim=output_dim
        self.out_linear=nn.Linear(768,output_dim)
    
    def forward(self,input_ids,attn_mask):
        enc_out=self.bert_model(input_ids=input_ids,attention_mask=attn_mask)
        fin_vect=enc_out.last_hidden_state[:,1:,:]
        return fin_vect

class GNNtoGPT2(nn.Module):
    def __init__(self, gnn_encoder, text_encoder,gpt2_model, gpt2_hidden_dim):
        super(GNNtoGPT2, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.text_encoder=text_encoder
        self.gpt2_model = gpt2_model
        self.num_heads=1
        self.linear = nn.Linear(768+gnn_encoder.output_dim, gpt2_hidden_dim)
        self.mha = nn.MultiheadAttention(768, self.num_heads)

    def forward(self, node_indices, edge_index, edge_type, input_ids, attention_mask,text_ids,text_mask,batch,mask):
        # print("node indices size",node_indices.size())
        graph_embedding = self.gnn_encoder(node_indices, edge_index, edge_type,batch,mask)
        text_embedding=self.text_encoder(text_ids,text_mask)
        # print("graph tensor",graph_embedding.size(),"text tensor",text_embedding.size())
        # print(text_embedding.size(),graph_embedding.size())
        graph_embed_transpose=graph_embedding.transpose(0,1)
        text_embed_transpose=text_embedding.transpose(0,1)
        full_embedding=self.mha(query=graph_embed_transpose,key=text_embed_transpose,value=text_embed_transpose)
        full_embed_transpose=full_embedding[0].transpose(0,1)
        full_embed_transpose=full_embed_transpose.contiguous()
        # gpt2_embedding = self.linear(full_embedding).unsqueeze(0)
        padding_mask = input_ids != gpt2_tokenizer.pad_token_id
        input_ids[~padding_mask] = 0
        print("Shape of encoder_hidden_states:", full_embed_transpose.shape)
        print("Is encoder_hidden_states contiguous?", full_embed_transpose.is_contiguous())
        # inputs_embeds = self.gpt2_model.transformer.wte(input_ids)
        # inputs_embeds[:, 0] = gpt2_embedding

        outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask,encoder_hidden_states=full_embed_transpose)
        return outputs

def train(model,loader,optimizer,criterion,model_name):
    best_loss=0
    best_epoch=0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            # print(batch)
            optimizer.zero_grad()
            batch_node_indices = batch['graph'].x.to(DEVICE)
            batch_edge_index = batch['graph'].edge_index.to(DEVICE)
            batch_edge_type = batch['graph'].edge_type.to(DEVICE)
            batch_ids=batch['graph'].batch.to(DEVICE)
            # print("batch ids",batch_ids)
            batch_input_ids = batch['input_ids'].to(DEVICE)
            batch_attention_mask = batch['attention_mask'].to(DEVICE)
            batch_amr_ids=batch['amr_ids'].to(DEVICE)
            batch_amr_mask=batch['amr_mask'].to(DEVICE)
            graph_mask=batch['graph'].mask.to(DEVICE)
            out = model(batch_node_indices, batch_edge_index, batch_edge_type, batch_input_ids, batch_attention_mask, batch_amr_ids,batch_amr_mask,batch_ids,graph_mask)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch_input_ids[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss=total_loss / len(loader)
        if epoch==0:
            best_loss=avg_loss
            # model_fin_name=model_name+"_"+str(best_loss)+"_"+str(epoch)+".pt"
            # torch.save(model.state_dict(),model_fin_name)
            best_epoch=epoch
        elif best_loss>avg_loss:
            # print("saving model here")
            best_loss=avg_loss
            best_epoch=epoch
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")
    model_fin_name=model_name+"_"+str(best_loss)+"_"+str(best_epoch)+".pt"
    torch.save(model.state_dict(),model_fin_name)

input_dim = data_list[0].x.size(0)
hidden_dim = 768
num_relations = len(edge_label_map)
learning_rate = 0.01
epochs = 50

gnn_encoder = GNNEncoder(num_nodes=len(node_to_idx), embedding_dim=128, hidden_dim=hidden_dim, output_dim=768, num_relations=num_relations)
text_encoder=LinearizedGraphEnc(768)
model = GNNtoGPT2(gnn_encoder,text_encoder, gpt2_model, 768)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
train(model,loader,optimizer,criterion,"mha_align_graph_text")
