import sfanalysis.sfanalysis as sf
import os
import torch
from utils.utils import (
    parse_arguments,
    load_config,
    edge_index_to_gml,
    split_edge_index_by_label,
    get_data,
    undirected_edge_count,
    count_hetero_edges,
    nodes_by_group)

from datetime import datetime

now = datetime.now()

args = parse_arguments()
config = load_config(args.yaml_config_file)

data = get_data(config = config)
dataset_name = config['sfanalysis']['dataset_name']
num_nodes = data.x.size(0)
num_classes = int(data.y.max()) + 1

edges_per_class = split_edge_index_by_label(data.edge_index, data.y)

# precisaria de uma variavel node tag per class
node_tag_per_class = nodes_by_group(data.y)

classes, counts = torch.unique(data.y, return_counts=True)
nodes_per_class = {int(c.item()): int(n.item()) for c, n in zip(classes, counts)}

number_edges_per_class = {
        int(c): undirected_edge_count(ei) for c, ei in edges_per_class.items()
    }

number_hetero_edges = count_hetero_edges(data.edge_index, data.y)

# Creating gml files per class
for label, ei in edges_per_class.items():
    if len(dataset_name.split(':')) == 1:
            dataset_name = dataset_name + ':' + dataset_name

    print(ei)
    
    if config['sfanalysis']['mimic_data']:
            gml_dir = f'datasets/{config["sfanalysis"]["synthetic_gen"]}/{dataset_name}_n_{config["sfanalysis"]["n"]}_m_{config["sfanalysis"]["m"]}/{now}/gmls/'
    else:
            gml_dir = f'datasets/{dataset_name.split(":")[1]}/{now}/gmls/'

    if not os.path.exists(gml_dir):
            os.makedirs(gml_dir)
    
    edge_index_to_gml(ei, nodes = node_tag_per_class[label], path = gml_dir + f'{dataset_name}_{label}.gml')

# gml file for the complete data

if config['sfanalysis']['mimic_data']:
    edge_index_to_gml(data.edge_index, path = f'datasets/{config["sfanalysis"]["synthetic_gen"]}/{dataset_name}_n_{config["sfanalysis"]["n"]}_m_{config["sfanalysis"]["m"]}/{now}/gmls/' + f'{dataset_name}_complete.gml')

else:
    edge_index_to_gml(data.edge_index, path = f'datasets/{dataset_name.split(":")[1]}/{now}/gmls/complete_{dataset_name}_complete.gml')

if config['sfanalysis']['mimic_data']:
        deg_dir = f'datasets/{config["sfanalysis"]["synthetic_gen"]}/{dataset_name}_n_{config["sfanalysis"]["n"]}_m_{config["sfanalysis"]["m"]}/{now}/degseqs/'

else: 
    deg_dir = f'datasets/{dataset_name.split(":")[1]}/{now}/degseqs/'

if not os.path.exists(deg_dir):
    os.makedirs(deg_dir)

deg_df = sf.write_degree_sequences(gml_dir, deg_dir)
analysis_df = sf.analyze_degree_sequences(deg_dir, deg_df)
hyps_df = sf.categorize_networks(analysis_df)

# save dataframes

if config['sfanalysis']['mimic_data']:
    main_path = f'outputs/{config["sfanalysis"]["synthetic_gen"]}/{dataset_name}/'

else:
    main_path = f'outputs/{dataset_name}/'

if not os.path.exists(main_path):
    os.makedirs(main_path)

if config['sfanalysis']['mimic_data']:
    hyps_df.to_csv(main_path + f'network_category_n_{config["sfanalysis"]["n"]}_m_{config["sfanalysis"]["m"]}_{now}.csv')

else:
     hyps_df.to_csv(main_path + f'network_category_{now}.csv')