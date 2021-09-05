import numpy as np
import pickle

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = int(np.sum(gnd))
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    print(num_query)
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap
def pr_curve(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    dim = np.shape(rB)
    bit = dim[1]
    all_ = dim[0]
    precision = np.zeros(bit + 1)
    recall = np.zeros(bit + 1)
    num_query = queryL.shape[0]
    num_database = retrievalL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        all_sum = np.sum(gnd).astype(np.float32)
        # print(all_sum)
        if all_sum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        # print(hamm.shape)
        ind = np.argsort(hamm)
        # print(ind.shape)
        gnd = gnd[ind]
        hamm = hamm[ind]
        hamm = hamm.tolist()
        # print(len(hamm), num_database - 1)
        max_ = hamm[num_database - 1]
        max_ = int(max_)
        t = 0
        for i in range(1, max_):
            if i in hamm:
                idd = hamm.index(i)
                if idd != 0:
                    sum1 = np.sum(gnd[:idd])
                    precision[t] += sum1 / idd
                    recall[t] += sum1 / all_sum
                else:
                    precision[t] += 0
                    recall[t] += 0
                t += 1
        # precision[t] += all_sum / num_database
        # recall[t] += 1
        for i in range(t,  bit + 1):
            precision[i] += all_sum / num_database
            recall[i] += 1
    true_recall = recall / num_query
    precision = precision / num_query
    print(true_recall)
    print(precision)
    return true_recall, precision
def CalcNDCG_N(N, qB, rB, queryL, retrievalL):
    num_q = qB.shape[0]
    a_NDCG = 0.0
    NDCG = 0.0
    for i in range(num_q):
        DCG = 0.0
        max_DCG = 0.0
        sim = (np.dot(queryL[i, :], retrievalL.transpose())).astype(np.float32)
        # qL = np.sum(queryL[i,:]).astype(np.float32)
        # rL = np.sum(retrievalL, axis=1).astype(np.float32)
        # L = np.power(qL * rL, 0.5).astype(np.float32)
        # sim = sim / L
        hamm = calc_hammingDist(qB[i, :], rB)
        ind = np.argsort(hamm)
        sim_sort = np.argsort(sim)
        for k in range(N):
            gain = 2 ** sim[ind[k]] - 1
            gain_max = 2 ** sim[sim_sort[- k - 1]] - 1
            log = np.log2(k + 2)
            DCG += gain / log
            max_DCG += gain_max / log
        NDCG += DCG / max_DCG
    a_NDCG = NDCG / num_q
    return a_NDCG
if __name__=='__main__':

    # B_L = {'Qi': qB_img, 'Qt': qB_txt,
    #        'Di': rB_img, 'Dt': rB_txt,
    #        'Unified': rB,
    #        'retrieval_L': database_labels.numpy(), 'query_L': test_labels.numpy()}
    for i in [16, 32, 48, 64]:
        with open('nus21_code_' + str(i) + '.pkl', 'rb') as f:
            B_L = pickle.load(f)
        qB_img = B_L['qB']
        rB = B_L['dB']
        test_labels = B_L['query_L']
        database_label_single = (B_L['retrieval_L'])
        num_database = database_label_single.shape
        print(qB_img.shape)
        print(rB.shape)
        print(test_labels.shape)
        print(test_labels[0, :])
        print(rB[0, :])
        print(num_database)
        map = calc_topMap(qB_img, rB, test_labels, database_label_single, 5000)
        print(map)
        ndcg = CalcNDCG_N(5000, qB_img, rB, test_labels, database_label_single)
        print(ndcg)
        recall, precision = pr_curve(qB_img, rB, test_labels, database_label_single)
        with open('./pr/pr_DPSH_nus21' + str(i) + 'bits.pkl', 'wb') as f:
            pickle.dump({'recall': recall, 'precision': precision}, f)
    # import matplotlib.pyplot as plt
    # ln7 = plt.plot(recall, precision, marker='|', linestyle='-', label="CDQ", color='magenta', )
    # # plt.legend(['PSHUH', 'SSAH', 'CDMH', 'SePH', 'SCM', 'DLFH', 'CDQ'], loc='upper right')
    # plt.xlim(0, 1)  # 限定纵轴的范围
    # # plt.ylim(0.3, 0.8)  # 限定纵轴的范围
    # plt.xticks(list(np.linspace(0, 1, 11)), size=15)
    # plt.yticks(size=15)
    # plt.xlabel("recall", size=18)  # X轴标签
    # plt.ylabel("precision", size=18)  # Y轴标签
    # plt.title("I2T@32bits on IAPR TC-12", size=20)  # 标题
    # # pdf.savefig()
    # plt.show()