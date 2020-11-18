
from Carttree import creat_Carttree
from ID3tree import creat_ID3tree
from C45tree import creat_C45tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz


if __name__ == '__main__':
    ## 导入数据集
    df_data = pd.read_table('xigua.txt', sep=',', encoding='gbk')
    dataset = df_data.drop(columns=['编号'])
    dataset_id3 = df_data.drop(columns=['编号', '含糖率', '密度'])

    ## 用自己写的代码进行测试
    ## 利用西瓜数据集生成分类决策树
    print('——————————————————MY ID3tree————————————————————')
    my_ID3tree = creat_ID3tree(dataset_id3, dataset_id3.columns, '好瓜', alpha=0.001, strRoot="-", strRootAttri="-")
    print('\n\n')
    print('——————————————————MY C45tree————————————————————')
    my_C45tree = creat_C45tree(dataset, dataset.columns, '好瓜', alpha=0.001, strRoot="-", strRootAttri="-")
    print('\n\n')
    print('——————————————————MY Carttree————————————————————')
    my_Carttree = creat_Carttree(dataset, dataset.columns, '好瓜', sample_threshold = 0, gini_threshold = 0, typeof = 'classification', strRoot="-", strRootAttri="-")

    ## 使用sklearn包来实现决策树

    ## 数据转化
    data_id3_convert = dataset_id3.copy()
    for col in data_id3_convert.columns:
        a = list(set(data_id3_convert[col]))
        for i in range(len(a)):
            data_id3_convert[col][data_id3_convert[col] == a[i]] = i


    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(data_id3_convert.iloc[:,:-1], data_id3_convert['好瓜'].astype('int'))
    # 生成可视化图
    dot_data = export_graphviz(tree, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("iris")