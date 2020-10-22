import os
import time
import numpy as np

import dgl
from dgl import backend as F

import argparse
import torch as th

def create_range_partition_policy(args):
    """Create RangePartitionBook and PartitionPolicy
    """
    node_map = F.tensor(np.array([args.graph_size, 2*args.graph_size], np.int64))
    edge_map = F.tensor([1,2])

    gpb = dgl.distributed.graph_partition_book.RangePartitionBook(part_id=args.machine_id,
                                                                  num_parts=args.num_machine,
                                                                  node_map=node_map,
                                                                  edge_map=edge_map)

    policy = dgl.distributed.PartitionPolicy(policy_str='node',
                                             partition_book=gpb)
    return policy, gpb 

def create_data(args):
    """Create data hold by server nodes
    """
    data = F.zeros((args.graph_size, args.dim), F.float32, F.cpu())
    return data

def start_server(args):
    print("create data...")
    data = create_data(args)
    print("Create data done.")
    kvserver = dgl.distributed.KVServer(server_id=args.server_id,
                                        ip_config='ip_config.txt',
                                        num_servers=num_servers,
                                        num_clients=1)

    policy, gpb = create_range_partition_policy(args)

    kvserver.add_part_policy(policy)

    if kvserver.is_backup_server():
        kvserver.init_data(name='data', policy_str='node')
    else:
        kvserver.init_data(name='data', policy_str='node', data_tensor=data)

    server_state = dgl.distributed.ServerState(kvserver, None, None)

    dgl.distributed.start_server(server_id=args.server_id,
                                 ip_config='ip_config.txt',
                                 num_servers=args.num_servers,
                                 num_clients=1,
                                 server_state=server_state)

def start_client(args):
    os.environ['DGL_DIST_MODE'] = 'distributed'
    policy, gpb = create_range_partition_policy(args)
    print("create data...")
    data = create_data(args)
    print("Create data done.")
    dgl.distributed.initialize(ip_config='ip_config.txt', num_servers=args.num_server)
    kvclient = dgl.distributed.KVClient(ip_config='ip_config.txt', num_servers=args.num_server)
    kvclient.map_shared_data(partition_book=gpb)

    #################################### local fast-pull ####################################

    
    if args.machine_id == 1:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)

    start = time.time()
    for _ in range(100):
        res = kvclient.pull(name='data', id_tensor=id_tensor)
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Local fast-pull Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))
    
    

    #################################### remote fast-pull ####################################

    
    if args.machine_id == 0:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
        id_tensor = id_tensor + args.graph_size
    else:
        id_tensor = np.random.randint(args.graph_size, size=args.data_size)
    id_tensor = F.tensor(id_tensor)

    start = time.time()
    for _ in range(100):
        res = kvclient.pull(name='data', id_tensor=id_tensor)
    end = time.time()
    total_bytes = (args.data_size*(args.dim+2)*4)*100*args.num_client/2
    print("Remote fast-pull Throughput (MB): %f" % (total_bytes / (end-start) / 1024.0 / 1024.0))

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--num_server', type=int, default=2,
                          help='Number of server on each machine.')
        self.add_argument('--machine_id', type=int, help="machine ID.")
        self.add_argument('--server_id', type=int, help='server_id')
        self.add_argument('--data_size', type=int, default=100000,
                          help='data_size of each machine.')
        self.add_argument('--dim', type=int, default=100,
                          help='dim of each data.')
        self.add_argument('--graph_size', type=int, default=1000000,
                          help='total size of the graph.')

if __name__ == '__main__':
    args = ArgParser().parse_args()

    if args.server_id == -1:
        start_client(args)
    else:
        start_server(args)