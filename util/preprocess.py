# preprocess data
import numpy as np
import re


def get_most_common_features(target, all_features, max = 3, min = 3):
    res = []
    main_keys = target.split('_')

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res

def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2
    
    for i in range(depth):        
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []
            
            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map


def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:#将数据根据特征标签加入res并转换成列表
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0]) #读取时间序列长度t

    if type(labels) == int:#根据情况（是否是有监督）
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res

def build_loc_net(struc, all_features, feature_map=[]): #输入： 图的连接关系、所有图上的节点/特征及其连接关系、图上的特征

    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items(): #将多元时间序列中的节点关系连接进行循环
        if node_name not in all_features: #以防万一没有提取到所有特征，这边应该是一个补充
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name) #这边应该也是由于特征提取不全做的一个补充操作，最后以struc的节点为准
        
        p_index = index_feature_map.index(node_name) #找到node_name在index_feature_map中的索引位置
        for child in node_list: 
            if child not in all_features:#检查nodelist是否包含了全部特征
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')#检查nodelist是否包含了全部节点
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child) #对于每个子（连接）节点输出其在map中的索引
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index) #返回的第一部分是子连接的索引 nx(n-1)
            edge_indexes[1].append(p_index) #返回的第二部分是每个dot的索引nx(x-1)
        

    
    return edge_indexes