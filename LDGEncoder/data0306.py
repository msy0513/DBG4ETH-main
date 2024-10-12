import os
import csv
import numpy as np
import random
import scipy.sparse as sp



def normal_time(links):
    times = np.zeros(len(links))
    for i in range(len(links)):
        times[i] = int(links[i][3])
    max_time = np.max(times)
    min_time = np.min(times)
    for i in range(len(links)):
        links[i][3] = (int(links[i][3]) - min_time)/(max_time - min_time)
    links = sorted(links, key=lambda x: x[3])
    return links


def multi2single(links):
    single_links = []
    link_dict = {}

    for link in links:
        node_pair = (link[0], link[1])
        link_info = link[2:]

        if node_pair in link_dict:
            link_dict[node_pair]['sum'] += float(link_info[0])
            link_dict[node_pair]['infos'].append(link_info)
        else:
            link_dict[node_pair] = {'sum': float(link_info[0]), 'infos': [link_info]}

    for node_pair, data in link_dict.items():
        data['infos'].sort(key=lambda x: float(x[-1]))
        timestamp = data['infos'][-1][-1]
        single_links.append([node_pair[0], node_pair[1], data['sum'], timestamp])

    return single_links


def get_subgraph1(node, nodes, links, dict, path, p):
    with open(path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            from_node = row["from"]
            to_node = row["to"]
            # if from_node == node:
            #     dict[from_node] = p
            # else:
            #     if from_node not in dict:
            #         dict[from_node] = row["isError"]
            # if to_node == node:
            #     dict[to_node] = p
            # else:
            #     if to_node not in dict:
            #         dict[to_node] = row["isError"]

            trans = [from_node, to_node, row["value"], float(row["timestamp"])]
            if from_node not in nodes:
                nodes.append(from_node)
            if to_node not in nodes:
                nodes.append(to_node)
            links.append(trans)


def get_subgraph2(node, nodes, links, dict, path, p):
    for root, dirs, files in os.walk(path):
        for filepath in files:
            csv_path = os.path.join(root, filepath)
            with open(csv_path) as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    from_node = row["from"]
                    to_node = row["to"]
                    # if from_node == node:
                    #     dict[from_node] = p
                    # else:
                    #     if from_node not in dict:
                    #         dict[from_node] = row["isError"]
                    # if to_node == node:
                    #     dict[to_node] = p
                    # else:
                    #     if to_node not in dict:
                    #         dict[to_node] = row["isError"]

                    trans = [from_node, to_node, row["value"], float(row["timestamp"])]
                    if from_node not in nodes:
                        nodes.append(from_node)
                    if to_node not in nodes:
                        nodes.append(to_node)
                    links.append(trans)


def dynamic_G(node, label, features, max_num):
    nodes = []
    links = []
    Node = []
    Label = []
    node_num = []
    adj = []
    fea = []
    dict = {}
    
    if label == 0:
        path1 = '/datasets/Exchange first-order nodes/' + node + '.csv'
        path2 = '/datasets/Exchange second-order nodes/' + node
        p = 0
    elif label == 1:
        path1 = '/datasets/ICO Wallets first-order nodes/' + node + '.csv'
        path2 = '/datasets/ICO Wallets second-order nodes/' + node
        p = 1
    elif label == 2:
        path1 = '/datasets/Mining first-order nodes/' + node + '.csv'
        path2 = '/datasets/Mining second-order nodes/' + node
        p = 2
    else:
        path1 = '/datasets/Phishing first-order nodes/' + node + '.csv'
        path2 = '/datasets/Phishing second-order nodes/' + node
        p = 3

    # get all first-order and second-order trans and nodes
    get_subgraph1(node, nodes, links, dict, path1, p)
    get_subgraph2(node, nodes, links, dict, path2, p)

    links = normal_time(links)
    dy_links = []
    dy_links_single = []
    # slice all trans to ten
    for i in range(10):
        dy_links.append([])
        for link in links:
            if link[3] < (i + 1) * 0.1 and link[3] >= (i) * 0.1:
                dy_links[i].append(link)

    # sum the trans amount with same from and to
    for index in range(len(dy_links)):
        if len(dy_links[index]) > 0:
            dy_links_single.append(sorted(multi2single(dy_links[index]), key=lambda x: x[3], reverse=True)[:max_num])
        else:
            dy_links_single.append([])

    # record nodes in snapshots
    dy_nodes = []
    for links in dy_links_single:
        slice_nodes = []
        for link in links:
            if link[0] not in slice_nodes:
                slice_nodes.append(link[0])
            if link[1] not in slice_nodes:
                slice_nodes.append(link[1])
        dy_nodes.append(slice_nodes)

    All_links = []
    for i in range(len(dy_links_single)):
        All_links += dy_links_single[i]
    # make node the first one in Node
    Node.append(node)
    for l in All_links:
        if l[0] not in Node:
            Node.append(l[0])
            # Label.append(int(dict[l[0]]))
        if l[1] not in Node:
            Node.append(l[1])
            # Label.append(int(dict[l[1]]))
    max_node = len(Node)
    print(max_node)
    
    # record the number of no_zero snapshots
    no_zero = 0
    for index in range(len(dy_links_single)):
        adj.append([])
        fea.append([])
        adj[index] = np.zeros((max_node, max_node), dtype=np.float32)
        fea[index] = np.zeros((max_node, 15), dtype=np.float32)

        if len(dy_links_single[index]) > 0:
            no_zero += 1
            # record feature
            for i, nd in enumerate(dy_nodes[index]):
                if nd in features:
                    fea[index][i] = features[nd]
            # record adj
            for l in dy_links_single[index]:
                # store value in adj
                adj[index][Node.index(l[0]), Node.index(l[1])] = l[2]

        # fea[index][0] = feature
        # if len(dy_links_single[index]) > 0:
        #     no_zero += 1
        #     for l in dy_links_single[index]:
        #         # store value in adj
        #         adj[index][Node.index(l[0]), Node.index(l[1])] = l[2]
    return adj, fea, max_node, len(links), no_zero

def load_graph(file_path):
    Data = np.load(file_path, allow_pickle=True)
    return {
        'Adj': Data['adj'],
        'Fea': Data['fea'],
        'Label': Data['label'],
        'Batch_num_nodes':Data['batch_num_nodes'],
        'Link_num': Data['link_num'],
        'No_zero': Data['no_zero'],
        'Nodes': Data['hash']
    }
def con_dynamic_Gset(save_path,batch_size, max_link, max_n=None):
    if os.path.exists(save_path+ "_val.npz"):
        print('G_set has already exists!')

        train_Graph = load_graph(save_path + "_train.npz")

        test_Graph = load_graph(save_path + "_test.npz")

        val_Graph = load_graph(save_path + "_val.npz")
    else:
        print('Constrct Graph Set......')
        nodes = []
        node_feature = []

        #
        with open(r"\datasets\exchanghe\phishing_node1.csv") as feature_file:
            reader = csv.reader(feature_file)
            next(reader)
            for row in reader:
                nodes.append(row[0])
                node_feature.append(row)
       
        rows=node_feature

        # random.shuffle(rows)
        # load features
        features = {}
        with open(r"\datasets\node_feature.csv", "r") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                features[row[0]] = [float(item) for item in row[1:]]


        split_index=280
        val_index=140
        # 0,1
        train_rows = rows[:split_index]
        # 2
        val_rows = rows[split_index:split_index + val_index]
        # 3
        test_rows = rows[split_index + val_index:]


        train_Graph = construct_graph(train_rows, batch_size,max_link, max_n, features)#276
        np.savez(save_path + "_train", adj=train_Graph['Adj'], fea=train_Graph['Fea'], label=train_Graph['Label'],batch_num_nodes=train_Graph['Batch_num_nodes'],
                 link_num=train_Graph['Link_num'], no_zero=train_Graph['No_zero'], hash=train_Graph['Nodes'],)
        #
        test_Graph = construct_graph(test_rows,batch_size, max_link, max_n, features)#137
        np.savez(save_path + "_test", adj=test_Graph['Adj'], fea=test_Graph['Fea'], label=test_Graph['Label'],batch_num_nodes=test_Graph['Batch_num_nodes'],
                 link_num=test_Graph['Link_num'], no_zero=test_Graph['No_zero'], hash=test_Graph['Nodes'])


        val_Graph = construct_graph(val_rows,batch_size, max_link, max_n,features)#47
        np.savez(save_path + "_val", adj=val_Graph['Adj'], fea=val_Graph['Fea'], label=val_Graph['Label'],batch_num_nodes=val_Graph['Batch_num_nodes'],
                 link_num=val_Graph['Link_num'], no_zero=val_Graph['No_zero'], hash=val_Graph['Nodes'])

        # np.savez(save_path, adj=Graph['Adj'], fea=Graph['Fea'], label=Graph['Label'],
        #           link_num=Graph['Link_num'],no_zero=Graph['No_zero'], hash=Graph['Nodes'])
    #
    # max_num_nodes = max(
    #     [train_Graph['Adj'][0][0].shape[0], test_Graph['Adj'][0][0].shape[0], val_Graph['Adj'][0][0].shape[0]])
    # print("Max Number of Nodes:", max_num_nodes)
    # return Graph

    return {
        'train_Graph': train_Graph,
        'test_Graph': test_Graph,
        'val_Graph': val_Graph,
        # 'max_num_nodes': max_num_nodes
    }


def construct_graph(rows, batch_num,max_link, max_n,features):
    # 保存正负例样本所有节点、标签和特征
    Nodes = []
    Labels = []
    # features = []

    Link_num = []
    No_zero = []

    target_labels = ['Exchange', 'ICO Wallets', 'Mining', 'Phishing']

    for row in rows:
        Nodes.append(row[0])
        # features.append(row[2:16])  
        Labels.append(target_labels.index(row[16]))

    A = []
    L = []
    F = []
    N = []
    Adj = []
    Label = []
    Fea = []
    Batch_num_nodes = []
    if max_n is None:
        max_num = 0
    else:
        max_num = max_n

    # construct graphs
    num = 0
    
    for node in Nodes:
        print(node)
        if num == 0:
            A.append([])
            L.append([])
            F.append([])
            N.append([])
            Adj.append([])
            Fea.append([])
        if num < batch_num - 1:
            num += 1
        else:
            num = 0
        # adj, fea, node_num, link_num, no_zero = dynamic_G(node, Labels[Nodes.index(node)], features[Nodes.index(node)],
        #                                                   max_link)

        adj, fea, node_num, link_num, no_zero = dynamic_G(node, Labels[Nodes.index(node)], features,
                                                          max_link)
        
        Link_num.append(link_num)
        
        No_zero.append(no_zero)
        batch_node_num = []
       
        for i in range(len(adj)):
            if max_n is None:
                if node_num > max_num:
                    max_num = node_num
            if node_num >= max_num:
                adj[i] = adj[i][0:max_num, 0:max_num]
                fea[i] = fea[i][0:max_num]
                node_num = max_num
                batch_node_num.append(max_num)
            else:
                batch_node_num.append(node_num)
            # make it csr_matrix
            adj[i] = sp.csr_matrix(adj[i])
            fea[i] = sp.csr_matrix(fea[i])
        A[-1].append(adj)
        F[-1].append(fea)
        # print('batch_node_num'+str(batch_node_num))
        # msy
        # only record label with 0 or 1(negetive samples)
        if Labels[Nodes.index(node)] == 0:
            L[-1].append(1)
        else:
            L[-1].append(0)
        # L[-1].append(Labels[Nodes.index(node)])
        N[-1].append(batch_node_num)

    for batch in range(len(Adj)):
        # for dy_time in range(len(adj)):
        for dy_time in range(batch_num):
            Adj[batch].append([])
            Fea[batch].append([])
   
    for i in range(len(Adj)):
       
        for j in range(batch_num):
           
            for n in range(len(adj)):
                Adj[i][j].append(np.zeros((max_num, max_num), dtype=np.float32))
                Fea[i][j].append(np.zeros((max_num, 15), dtype=np.float32))
                # print('Adj[i][j]'+str(len(Adj[i][j])))
                # print(Adj[i][j][n].shape)
                # print('A[i][j][n].shape'+str(A[i][j][n].shape))
                # print('A[i][j][n]' + str(A[i][j][n]))
                # print('A[i][j]' + str(A[i][j][n].shape[0]))
                l = A[i][j][n].shape[0]


                # l, _ = A[i][j][n].shape

                Adj[i][j][n][:l, :l] = A[i][j][n].todense()
                Fea[i][j][n][:l, :] = F[i][j][n].todense()
                Adj[i][j][n] = sp.csr_matrix(Adj[i][j][n])
                Fea[i][j][n] = sp.csr_matrix(Fea[i][j][n])
        Adj[i] = np.array(Adj[i])
        Fea[i] = np.array(Fea[i])
        Label.append(np.array(L[i]))
        Batch_num_nodes.append(np.array(N[i]))
        Batch_num_nodes[i] = np.array(Batch_num_nodes[i])

    # return the constructed graph
    return {
        'Adj': Adj,
        'Fea': Fea,
        'Label': Label,
        'Batch_num_nodes':Batch_num_nodes,
        'max_num': max_num,
        'Link_num': Link_num,
        'No_zero': No_zero,
        'Nodes': Nodes
    }


