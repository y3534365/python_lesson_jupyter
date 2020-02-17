import numpy as np
import pandas as pd

def cal_recommend_index_by_users(k , user_id ,item_id ,freq_matrix,user_similar_matrix):
    """函数功能: 传入用户ID , 和产品ID ,以及最相似的k个用户,返回该产品对该用户的推荐指数
    参数说明:
    k : 计算相似用户的个数
    user_id : 用户ID
    item_id : 产品ID
    freq_matrix : 用户对于产品的评分矩阵
    user_similar_matrix : 用户相似度矩阵
    """

    # 提取出该用户的评分向量
    user_id_action = freq_matrix[user_id]

    # 再提取出产品的评分向量
    item_id_action = freq_matrix[:,item_id ]

    # 提取出该用户和其他用户的相似度向量
    user_id_similar = user_similar_matrix[user_id]

    # 因为k = 2 所有要从相似度向量里面提取出最大的相似的两个用户来

    # [::-1] 是将向量翻转 ,变成从大到小, 我们要去的是 除了自身(最大的1)以外
    # 最大的两个数值对应的索引, 所以用索引 [1:k+1]
    # 把相似度最高的两个用户ID保存下来
    
    similar_index =  np.argsort(user_id_similar)[::-1][1:k+1]

    # 进行该用户对该产品评分的运算

    # 定义一个常量用于储存评分计算结果
    score = 0
    # 在定义一个变量用于储存用户的相似度之和
    weight = 0

    # 对这k个最相似的用户分别进行计算, 使用循环
    for similar_user_id in  similar_index:
        # 我们先判断,这个相似用户有没有对这个产品评分
        if item_id_action[similar_user_id] != 0:
        # 如果这个值不等于0 ,我们才需要进行下面的计算, 否则直接跳过就可以
            #  使用这个用户的我们计算用户的相似度, 乘以他对该产品的评分值
            # 将计算结果累加到score中去
            score += user_id_similar[similar_user_id] * item_id_action[similar_user_id]
            # 将用户的相似度进行累加
            weight += user_id_similar[similar_user_id]

    # 当上面的循环结束后, 也就把所有的用户评分相关计算完成了
    # 首先,我们判断 score是否等于0 , 代表相似用户都没有评价过这个产品,则推荐指数直接设置为 0 
    if score == 0:
        return 0
    else:
        # 计算最终推荐指数
        rec_index = score / weight
        return rec_index
    
    
def cal_recommend_index_matrix_by_users(freq_matrix,user_similar_matrix,k=2 ):
    """函数功能: 传入用户评分矩阵, 计算并返回用户对所有产品的推荐矩阵
    参数:
    freq_matrix: 用户评分矩阵
    返回值: 推荐系数矩阵"""
    predict_matrix = np.zeros_like(freq_matrix)

    # 写一个两层循环, 分别对predict_matrix 矩阵中的每一个位置的值进行遍历
    # predict_matrix.shape[0] 代表用户个数 , range()生成一个线性序列和这个长度相同
    for user_id in range(predict_matrix.shape[0]):
        # 对每行中的每一个值进行遍历
        for item_id in range(predict_matrix.shape[1]):
            # 首先判断原来这个用户有没有对这个产品评分过,如果有不需要计算,如果没有 值为0,则进行计算
            if freq_matrix[user_id,item_id ] == 0:
                # 如果为0 计算推荐指数,
                recommend_index= cal_recommend_index_by_users(k, user_id, 
                                item_id ,freq_matrix,user_similar_matrix)
                # 算出来之后, 把这个数值保存到矩阵中去
                predict_matrix[user_id , item_id] = recommend_index
    return predict_matrix




    

#构建一个物品的推荐
def cal_recommend_index_by_items(k , user_id,item_id,freq_matrix,item_similar_matrix):
    """函数功能: 传入用户ID , 和产品ID ,以及最相似的k个用户,返回该产品对该用户的推荐指数
    参数说明:
    k : 计算相似用户的个数
    user_id : 用户ID
    item_id : 产品ID
    freq_matrix : 用户对于产品的评分矩阵
    item_similar_matrix : 产品相似度矩阵
    """
    
    user_id_action = freq_matrix[user_id,:]      #用户user_id 对所有商品的行为评分  
    item_id_action = freq_matrix[:,item_id]      #物品item_id 得到的所有用户评分  
    

    item_id_similar = item_similar_matrix[item_id,:]      #商品item_id 对所有商品的相似度    
    similar_indexs = np.argsort(item_id_similar)[-(k+1):-1]  #最相似的k个物品的index（除了自己）
    
    #计算该用户的评分均值
    # item_id_action 该产品的评分向量 ,np.sum(item_id_action) 评分之和
    # item_id_action[item_id_action!=0].size 非0值的个数
    # 用这两个数相除, 得到的就是平均分数
    item_id_mean = np.sum(item_id_action)/item_id_action[item_id_action!=0].size
    
    # 定义一个常量用于储存评分计算结果
    score = 0
    # 在定义一个变量用于储存用户的相似度之和
    weight = 0
    
    # 对所有相似产品进行遍历 , 产品ID赋值为similar_item_id
    for similar_item_id in similar_indexs :
        # 判断用户对这个产品是否有评分
        if user_id_action[similar_item_id]!=0:
            # 提取这个产品的评分向量
            similar_item_id_vector = freq_matrix[:,similar_item_id]
            # 计算这个产品的评分平均值,和上面一样
            similar_item_id_mean = np.sum(similar_item_id_vector)/similar_item_id_vector[similar_item_id_vector!=0].size
            # 将用户评分 减去 均值  ,然后乘以 产品之间的相似度
            score += item_id_similar[similar_item_id]*(user_id_action[similar_item_id]-similar_item_id_mean)
            # 权重累加和, 添加绝对值, 因为去掉均值有可能是负的
            weight += abs(item_id_similar[similar_item_id])

    if score==0:  
        return 0
    else:
        # 注意公式的形式, 和公式一样的计算过程
        return item_id_mean + score/float(weight)
    








def get_recom( predict_matrix ,df_pivot, k = 2):
    """函数功能, 传入推荐指数计算矩阵, 以及透视表DataFrame, 计算出所有用户的推荐结果
    参数:
    predict_matrix : 推荐指数计算矩阵
    df_pivot: 构建的相似度表格
    k : 搜索前几个推荐物品
    返回值: 推荐结果
    """
    # 把这个推荐矩阵转换成DataFrame, 索引值使用之前的那个df_pivot的索引值
    re_df = pd.DataFrame(predict_matrix ,index=df_pivot.index , 
                 columns= df_pivot.columns)
    # 对多重索引的Series进行 reset_index , 就会把索引和列都变成DataFrame里面的数据
    re_df_2 = re_df.stack().reset_index()
    # 把上面的列名0 改成 推荐指数
    re_df_2.rename(columns={0:"推荐指数"} ,inplace=True)
    # 保存分组结果, 但是先不进行合并
    re_df_grouped = re_df_2.groupby(by='用户ID')
    # 定义函数功能, 参数n代表要提取几个东西
    # 第一个参数分组结果
    def get_topn(group, n):
        # group 就代表了每一组数据
        # 计算出每一组数据中的最大的两个数据的id 和指数
        # 首先对每组数据进行排序, 排序后切片出前两个最大的值
        r = group.sort_values(by='推荐指数', ascending=False)[:n]
        return r
    # 我们使用 分组中的apply功能, 这个功能可以自定义函数
    topn = re_df_grouped.apply(get_topn, n = 2)
    # 上面的结果多了一个无用的索引, 把它删除掉
    # 把删除后的新的索引传递给这个表
    topn.index = topn.index.droplevel(1)
    topn.drop( columns='用户ID',inplace = True)
    return topn



    
    