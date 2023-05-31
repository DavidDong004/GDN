from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def get_full_err_scores(test_result, val_result):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result) #转换测试集和验证集的结果并赋值

    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1] #获取节点数量/特征数量

    labels = np_test_result[2, :, 0].tolist() #获取测试标签

    for i in range(feature_num): #对每个特征进行迭代，计算出相应误差评分
        test_re_list = np_test_result[:2,:,i]
        val_re_list = np_val_result[:2,:,i] #获取每个特征的测试结果和验证结果，包含两部分，0为预测值，1为真实值

        scores = get_err_scores(test_re_list, val_re_list)
        normal_dist = get_err_scores(val_re_list, val_re_list)#获取测试集和验证集的误差评分

        if all_scores is None: #如果评分为空，则赋值
            all_scores = scores
            all_normals = normal_dist 
        else: #如果不为空，则按行合并
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores



def get_err_scores(test_res, val_res): #获取误差函数
    test_predict, test_gt = test_res 
    val_predict, val_gt = val_res #获得测试集和验证集的参数

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    )) #计算预测值和测试值之间的绝对差值，论文式(11)
    epsilon=1e-2 

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon) #计算错误评分，论文式(12)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3 #平滑窗口大小 
    for i in range(before_num, len(err_scores)): #遍历数组，计算平滑窗口平均值
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    
    return smoothed_err_scores #返回平滑窗口后的误差



def get_loss(predict, gt):
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    print('total_err_scores', total_err_scores.shape)
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1): #获取最佳性能数据？

    total_features = total_err_scores.shape[0] #获取总特征数

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]  #获取在每个时间点上误差最大值所在的时间序列编号，论文中公式(13)

    total_topk_err_scores = [] #获取前k个特征的错误得分
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0) #对前k个数据进行求和，得到其最终的错误得分

    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True) #计算出

    th_i = final_topk_fmeas.index(max(final_topk_fmeas)) #找到最大f1分数所对应的索引
    thresold = thresolds[th_i]  #获取最大f1分数所对应的阈值

    pred_labels = np.zeros(len(total_topk_err_scores)) 
    pred_labels[total_topk_err_scores > thresold] = 1 #错误得分大于阈值位置为1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i]) #将真实值和预测值转换为整数便于判断

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores) #计算精确度召回率和ROC

    return max(final_topk_fmeas), pre, rec, auc_score, thresold

