import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# ## 関数群

# qr分解 numpy.linalg.qr確認用
def qr_out(A):
    orth_qs = np.array([])
    qs = np.array([])
    for i in range(A.shape[1]):
        # 1列目はそのまま直交化ベクトルqとなる
        if i == 0:
            orth_q = A[:,i]
            # 正規化
            q = orth_q / np.linalg.norm(orth_q)
            orth_qs = np.append(orth_qs, orth_q).reshape(i + 1, len(A[:,i]))
            qs = np.append(qs, q).reshape(i + 1, len(A[:,i]))
        else:
            orth_q = A[:,i] - np.array([qs[q].T @ A[:,i] * qs[q] for q in range( len(qs) )]).sum(axis=0)
            # print(orth_q)
            q = orth_q / np.linalg.norm(orth_q)
            orth_qs = np.append(orth_qs, orth_q).reshape(i + 1, len(A[:,i]) )
            qs = np.append(qs, q).reshape(i + 1, len(A[:,i]) )
    orth_qs = orth_qs.T
    qs = qs.T
    
    # QR分解のQを作ろう
    R = np.zeros((A.shape[0], A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                R[i][i] = np.linalg.norm(orth_qs[:,i])
            elif i < j:
                R[i][j] = qs[:,i].T @ A[:,j]
            elif i > j:
                R[i][j] = 0
    return qs, R

# 選択した次数までのすべての値を算出する
def return_element(S, m, N):
    # 選択した次数までをすべて取得
    sigma_estimates = []
    aics = []
    ar_coef = []
    for i in range(0, m + 1):
        Z = S[:i, :i]
        Zt = np.linalg.inv(Z)
        y_constant = S[:,i][i]
        y_full_vector = S[:,m][:m + 1]
        # print(y_full_vector)
        y = S[:, i][:i]
        yt = y.reshape(len(y), 1)
        # 回帰係数
        a = Zt @ yt
        # 残差分散推定値
        sigma_estimate = np.array([s**2 for s in y_full_vector[i : m+1]]).sum() / (N - m)
        # AIC
        aic_score = (N - m) * (np.log(2 * np.pi * sigma_estimate) + 1) + (2 * (i + 1))
        sigma_estimates.append(sigma_estimate)
        aics.append(aic_score)
        ar_coef.append(a)
    return pd.DataFrame({'sigma_estimate':sigma_estimates,
                        'aic_score': aics}), ar_coef

# Rのモデルは予めずらしてから分けてARFITしている
# arfitにはなのでずらし済みの行列を入れるべき
# ずらし済みの行列を処理して前のarfitと同じ変数たちを定義するClassを作る
class arfit_m:
    def __init__(self, data_matrix, lag, *args):
        self.data_matrix = data_matrix
        self.lag = lag
        # *argsに前の上三角行列、前の行列が入っている場合
        if args:
            self.pre_S = args[0]
            self.merged_matrix = args[1]
            self.Q, self.S = qr_out(np.vstack((self.pre_S, self.data_matrix)))
            self.result_df, self.ar_coef = return_element(self.S, self.lag, len(self.merged_matrix) + len(self.data_matrix) + self.lag)
        # 入っていない場合
        else:
            self.Q, self.S = qr_out(self.data_matrix)
            # 各種推定値
            self.result_df, self.ar_coef = return_element(self.S, self.lag, len(self.data_matrix) + self.lag)
        # 変数として返す
        self.minimum_aic = min(self.result_df['aic_score'])


# ## 局所定常ARモデル
# - 分割するサイズの最初単位L(50なら50ずつ区切ってデータを見ていってくっつけるかくっつけないかをAICで測る)
#   - 割り切れないサンプルは捨てる？
#   - 余すと最高次数mが適用できなくなる可能性がある
# - 各モデルの最高次数m
# - Lに対して当てはめられるような最高次数mにする必要がある(L = 50なら m = np.sqrt(L)とかでもいい)
# - 区間の個数k, 区間に対して使ったサンプルの位置
# - データを追加したときの方法について理解する必要がある

# ## 全体の流れ
# - データ全部とLとmを入れる
# - 区間に区切ってAICを算出していく
# - 区間同士をくっつけるときのAIC値を算出する時、上三角行列Sと追加するハウスホルダー変換対象の行列をくっつけてから、ハウスホルダー変換で上三角行列にする

# normal_ar パラメータとしてARモデル次数と誤差項
def normal_ar(data_array, local_length, max_arorder, plot=True):
    if np.sqrt(local_length) < max_arorder:
        raise Exception('最大次数に対して分割データ数が少なすぎます')
    df = pd.DataFrame(data_array, columns=['data'])
    # max_arorderをここで適用する
    for i in range(1, max_arorder + 1):
        df[i] = df['data'].shift(i)
    # ずらした分NaNが発生するので消す
    df.dropna(inplace=True)
    # 目的変数を後ろに持ってくる
    df = pd.concat([df[range(1, max_arorder + 1)], df['data']], axis=1)
    # local_length分だけ分ける あまったデータは削除
    matrix_list = [df.values[i:i+local_length] for i in range(0, len(df.values), local_length)]
    matrix_list = [x for x in list(map(lambda x : x if len(x) >= local_length else None, matrix_list)) if x is not None]
    # リザルト変数格納用
    AIC_0 = 0
    pre_S = 0
    merged_matrix = 0
    # 分割区間格納変数
    k_aic = np.array([])
    k_matrix = []
    # 分けられた行列を1つずつまわしていく
    for k in range(len(matrix_list)):
        print(f'{k + 1} 回目')
        # 初回の振る舞い
        if k == 0:
            model = arfit_m(matrix_list[k], lag=max_arorder) #最初の行列をARモデル推定
            S, minimum_aic = model.S, model.minimum_aic
            # 結合前のAIC_0, pre_S, merged_matrixを作成
            AIC_0 = minimum_aic ; pre_S = S ; merged_matrix = matrix_list[k]
            print(f'AIC_0 = {AIC_0}')
            print('初回')
        # 最後のループ
        elif k == len(matrix_list) - 1:
            model1 = arfit_m(matrix_list[k], lag=max_arorder)
            S1, minimum_aic1= model1.S, model1.minimum_aic
            model2 = arfit_m(matrix_list[k], max_arorder, pre_S, merged_matrix)
            S2, minimum_aic2 = model2.S, model2.minimum_aic
            AIC_D = AIC_0 + minimum_aic1
            AIC_P = minimum_aic2
            print(f'AIC_D = {AIC_D} AIC_P = {AIC_P}')
            if AIC_D < AIC_P:
                k_aic = np.append(k_aic, AIC_0)
                k_aic = np.append(k_aic, minimum_aic1) #最後なので分離して評価したminimum_aic1を格納
                k_matrix.append(merged_matrix)
                k_matrix.append(matrix_list[k]) # 最後なので分離して評価した行列を格納
                print('AIC_D を採択')
            elif AIC_P <= AIC_D:
                k_aic = np.append(k_aic, AIC_P) # 最後なので結合したAIC_Pを格納
                k_matrix.append(np.vstack((merged_matrix, matrix_list[k]))) # 最後なので結合した行列を格納
                print('AIC_P を採択')
        # それ以外
        else:
            # 分離した場合を評価
            model1 = arfit_m(matrix_list[k], lag=max_arorder)
            S1, minimum_aic1 = model1.S, model1.minimum_aic
            # 結合した場合を評価
            model2 = arfit_m(matrix_list[k], max_arorder, pre_S, merged_matrix)
            S2, minimum_aic2 = model2.S, model2.minimum_aic
            # 分離と結合の場合のAIC値を作成
            AIC_D = AIC_0 + minimum_aic1
            AIC_P = minimum_aic2
            print(f'AIC_D = {AIC_D} AIC_P = {AIC_P}')
        # 分離モデルと結合モデルでどちらが良いかをチェック
            if AIC_D < AIC_P:
                #aic配列には今までのAIC_0を格納
                k_aic = np.append(k_aic, AIC_0)
                # AIC_0はminimum_aic1に置き換える (分離の場合今までのAICは不要)
                AIC_0 = minimum_aic1
                # 行列格納リストには前の行列を格納
                k_matrix.append(merged_matrix)
                # merged_matrixに分離後の行列を指定
                merged_matrix = matrix_list[k]
                # pre_Sに分離後の上三角行列を指定
                pre_S = S1
                print('AIC_D を採択')
            elif AIC_P <= AIC_D:
                # 結合したAIC_P値をAIC_0として挿入
                AIC_0 = AIC_P
                # merged_matrixには結合した行列を挿入
                merged_matrix = np.vstack((merged_matrix, matrix_list[k]))
                # pre_Sを更新
                pre_S = S2
                print('AIC_P を採択')
    k_array = [m[:,-1] for m in k_matrix]
    print(f'分割数 = {len(k_array)}')

    # もともとのデータに対して区分けされたindex numberを取得
    pre_length = max_arorder
    length_array = []
    for array in k_array:
        length = len(array)
        length_array.append(np.arange(pre_length, pre_length + length))
        pre_length += length

    # plot=True なら結果を表示
    if plot:    
        plt.figure(figsize=(20, 12))
        plt.subplot(211)
        plt.plot(data_array)
        plt.title('input data')
        plt.subplot(212)
        for i in range(len(k_array)):
            plt.plot(length_array[i], k_array[i])
        plt.title('split result data')
        plt.show()

    return k_aic, k_array, length_array

