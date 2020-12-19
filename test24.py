# ID3 决策树算法（最终版本）

import pandas as pd
import math
from graphviz import Digraph

# 预处理数据表格，将表格中无用的信息（如：ID）去除
def pre_dataframe(df):
    total = df.shape[0]
    df_copy = df.copy()
    for column in df.columns:
        elements = []
        for index in df.index:              # 统计属性column的所有存在的value值
            elements.append(df.loc[index, column])
        elements_set = set(elements)
        if len(elements_set) == total:      # 对于某一个属性column，如果其值类型数等于数据条数，认为该属性没意义，删去
            df_copy.drop(column, axis=1, inplace=True)
    return df_copy

# 获取数据表中所有属性
def get_column_dict(df):
    columns_dict = {}
    for column in df.columns[:-1]:
        columns_dict[column] = []
        for index in df.index:
            if df.loc[index, column] not in columns_dict[column]:
                columns_dict[column].append(df.loc[index, column])
    return columns_dict

# 求取当前表格数据的“信息熵”
def get_info_entropy(df):
    df_results_series = df.loc[:, df.columns[-1]]       # 截取最后一列结果属性为Series向量表
    df_results_num = df_results_series.value_counts()   # 统计Series的信息
    info_entropy = 0
    total = df.shape[0]
    for label in df_results_num.index:                  # 计算“信息熵”
        ratio = (df_results_num[label] / total)
        info_entropy -= ratio * math.log(ratio, 2)
    return info_entropy

# 求取当前表格中每一个column的信息增益，并选出信息增益最大的属性
def get_best_column(df):
    best_column = 0
    best_info_gain = float('-inf')
    info_entropy = get_info_entropy(df)
    for column in df.columns[:-1]:                      # 遍历表格中除去“结果”的每一个属性
        temp_info_gain = info_entropy
        column_labels = dict()
        for index in df.index:      # 统计column属性每个value出现的次数
            if df.loc[index, column] not in column_labels.keys():
                column_labels[df.loc[index, column]] = 1
            else:
                column_labels[df.loc[index, column]] += 1
        for label in column_labels.keys():
            df_label = split_dataframe(df, column, label)   # 利用column属性value值划分表格
            info_entropy_label = get_info_entropy(df_label) # 获取划分后的表格的“信息熵”
            temp_info_gain -= (column_labels[label] / df.shape[0]) * info_entropy_label
        if temp_info_gain >= best_info_gain:          # 比较选取出信息增益最大的column属性
            best_column = column
            best_info_gain = temp_info_gain
    return best_column

# 使用属性column的值value对当前表格进行划分，某一行数据的column属性值满足value则留下，否则删除
def split_dataframe(df, column, value):
    df_copy = df.copy()
    for index in df.index:
        if df.loc[index, column] != value:      # 若该行数据column属性不等于value，则删除改行数据
            df_copy.drop(index, axis=0, inplace=True)
    df_copy.drop(column, axis=1, inplace=True)  # 将属性column从表格中删除
    return df_copy

# 递归建立决策树，形式为字典形式
def build_tree(df):
    global columns_dictionary
    result_dict = {}
    for index in df.index:      # 统计当前数据表中的结果属性的种类和数量
        if df.loc[index, df.columns[-1]] not in result_dict.keys():
            result_dict[df.loc[index, df.columns[-1]]] = 1
        else:
            result_dict[df.loc[index, df.columns[-1]]] += 1
    if len(result_dict) == 1:   # 如果结果属性全相同，则该叶子节点标记为该结果
        leaf_node = df.loc[df.index[0], df.columns[-1]]
        return leaf_node

    same_flag = 1       # 统计当前数据表中的所有属性的值是否相同
    label_columns = df.columns[:-1]
    for column in label_columns:
        temp = df.loc[df.index[0], column]
        for index in df.index:
            if df.loc[index, column] != temp:
                same_flag = 0
                break
        if same_flag == 0:
            break
    if same_flag == 1 or len(label_columns) == 0:   # 如果相同或者当前表中无属性，则将该叶子节点标记为数量最多的结果
        max_result = max(zip(result_dict.keys(), result_dict.values()))
        leaf_node = max_result[0]
        return leaf_node

    best_column = get_best_column(df)       # 获取“信息增益”最大的属性
    tree_node = {best_column: {}}           # 初始化决策树
    labels = []
    for index in df.index:
        if df.loc[index, best_column] not in labels:    # 获取最佳属性的所有值
            labels.append(df.loc[index, best_column])
    for label in columns_dictionary[best_column]:  # 利用最佳属性的值对数据表进行划分，并递归构建决策树
        if label in labels:
            df_label = split_dataframe(df, best_column, label)
            tree_node[best_column][label] = build_tree(df_label)
        else:
            max_result = max(zip(result_dict.keys(), result_dict.values()))
            tree_node[best_column][label] = max_result[0]
    return tree_node

# 利用graphviz包绘制决策树
def draw_tree(tree):
    global dot, node_num
    if type(tree) != dict:      # 如果该节点为叶子节点，则只画一个节点
        dot.node(chr(node_num), tree, fontname="Microsoft YaHei")
        return
    for key in tree.keys():
        temp = node_num
        dot.node(chr(node_num), key, fontname="Microsoft YaHei")    # 绘制该子树的根节点
        for key_edge in tree[key].keys():   # 遍历该根节点的每一条边
            node_num += 1
            if type(tree[key][key_edge]) != dict:   # 如果子节点为叶子节点
                dot.node(chr(node_num), tree[key][key_edge], fontname="Microsoft YaHei")
                dot.edge(chr(temp), chr(node_num), label=key_edge, fontname="Microsoft YaHei")
            else:   # 如果子节点不是叶子节点
                for key_edge_key in tree[key][key_edge].keys():
                    dot.node(chr(node_num), key_edge_key, fontname="Microsoft YaHei")
                    dot.edge(chr(temp), chr(node_num), label=key_edge, fontname="Microsoft YaHei")
            draw_tree(tree[key][key_edge])
    return


if __name__ == '__main__':
    io = 'D:\\西瓜数据集.xlsx'        # 文件路径
    # io = 'D:\\实验3训练数据-贷款申请.xlsx'
    dataframe = pd.read_excel(io)             # 读取excel表格数据，形式为pandas的DataFrame
    dataframe_prepared = pre_dataframe(dataframe)   # 预处理
    # print(dataframe_prepared)

    columns_dictionary = get_column_dict(dataframe_prepared)    # 获取数据表中所有属性

    decision_tree = build_tree(dataframe_prepared)  # 构造决策树
    # print(decision_tree)

    node_num = ord('A')
    dot = Digraph(name='The Decision Tree')         # 绘制决策树
    draw_tree(decision_tree)
    dot.view()
