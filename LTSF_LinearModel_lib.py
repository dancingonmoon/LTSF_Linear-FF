#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf


class series_decomp(tf.keras.layers.Layer):
    """
    Series decomposition block
    """

    def __init__(self, pool_size, name="series_comp", **kwargs):
        """
        input:
            pool_size: AveragePooling1D时,求平均值的窗口大小;如果pool_size-1为奇数,右边比左边多填1组;pool_size-1为偶数,则左右两边填充相等的组数;故而,pool_size设为奇数最佳
        output:
            res: sesonal component, shape与input一致
            moving_eman: trend cyclical component,shape与input一致
        """
        super(series_decomp, self).__init__(name=name, **kwargs)
        self.moving_avg = tf.keras.layers.AveragePooling1D(
            pool_size=pool_size, strides=1, padding="same"
        )  # 'same'时,经过查API文档,不是通过填充来补齐shape,而是求平均值的时候,求不同shape的平均值,即不足shape,平均值是剩余shape的平均值;所以,不需要补填充

    def call(self, x):
        moving_mean = self.moving_avg(x)  # shape remains unchanged
        res = x - moving_mean  # shape remains unchanged
        # print(moving_mean.shape)
        return res, moving_mean


class LTSF_DLinear(tf.keras.Model):
    """
    Decomposition-Linear
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        kernel_size=25,
        individual=False,
        name="LTSF_DLinear",
        **kwargs
    ):
        """
        seq_len: 输入序列长度;
        pred_len: 预测序列长度;
        channels: 输入序列的特征数;
        kernel_size: moving_avg(即:AveragePooling1D)时的pool_size(窗口大小)
        individual: 是否输入的特征每个独立预测,;
        """
        super(LTSF_DLinear, self).__init__(name=name, **kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = channels

        if self.individual:
            self.Linear_Seasonal = [
                tf.keras.layers.Dense(self.pred_len) for i in range(self.channels)
            ]
            self.Linear_Trend = [
                tf.keras.layers.Dense(self.pred_len) for i in range(self.channels)
            ]

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = tf.keras.layers.Dense(self.pred_len)
            self.Linear_Trend = tf.keras.layers.Dense(self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        # x: [Batch, Input length, Channel] -> [Batch,Channel,Input length]
        seasonal_init = tf.transpose(seasonal_init, perm=[0, 2, 1])
        # print('初始化:{}'.format(seasonal_init.shape))
        trend_init = tf.transpose(trend_init, perm=[0, 2, 1])

        if self.individual:
            seasonal_output = tf.zeros(
                [seasonal_init.shape[0], 1, self.pred_len],
                dtype=seasonal_init.dtype,
            ) #每个特征(channel)独立的进行Dense全连接(全部时序)
            trend_output = tf.zeros(
                [trend_init.shape[0], 1, self.pred_len],
                dtype=trend_init.dtype,
            )
            for i in range(self.channels):
                seasonal_output_tmp = self.Linear_Seasonal[i](
                    seasonal_init[:, i : i + 1, :]
                )  # i:i+1是为了保留这个维度,不消失
                # print('for循环seasonal_tmp:{}'.format(seasonal_output_tmp.shape))
                trend_output_tmp = self.Linear_Trend[i](trend_init[:, i : i + 1, :])
                seasonal_output = tf.concat(
                    [seasonal_output, seasonal_output_tmp], axis=1
                )
                # print('for循环,concat后:{}'.format(seasonal_output.shape))
                trend_output = tf.concat([trend_output, trend_output_tmp], axis=1)
            # print('for循环之后,{}'.format(seasonal_output.shape))
            seasonal_output = seasonal_output[:, 1:, :]  # concat将初始的0值拼接,这里舍去;
            trend_output = trend_output[:, 1:, :]
            # print('[:,1:,:]之后:{}'.format(seasonal_output.shape))
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        # print('perm恢复之前x:{}'.format(x.shape))
        return tf.transpose(x, perm=[0, 2, 1])  # to [Batch, Output length, Channel]

    
class LTSF_FF_DLinear(tf.keras.Model):
    """
    Decomposition-Linear
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        kernel_size=25,
        name="LTSF_DLinear",
        **kwargs
    ):
        """
        seq_len: 输入序列长度;
        pred_len: 预测序列长度;
        channels: 输入序列的特征数;
        kernel_size: moving_avg(即:AveragePooling1D)时的pool_size(窗口大小)
        """
        super(LTSF_FF_DLinear, self).__init__(name=name, **kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
        self.channels = channels
        
        self.Linear_Seasonal_channel0 = tf.keras.layers.Dense(channels*4,activation='relu')
        self.Linear_Seasonal_channel1 = tf.keras.layers.Dense(1)
        
        self.Linear_Seasonal_seqlen0 = tf.keras.layers.Dense(seq_len*4,activation='relu')
        self.Linear_Seasonal_seqlen1 = tf.keras.layers.Dense(pred_len)
        
        self.Linear_Trend_channel = tf.keras.layers.Dense(1)
        self.Linear_Trend_seqlen = tf.keras.layers.Dense(pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        
        seasonal_x = self.Linear_Seasonal_channel0(seasonal_init) #(Batch,Seq_len,channel*4)
        seasonal_x = self.Linear_Seasonal_channel1(seasonal_x) #(Batch,Seq_len,1)
        
        # x: [Batch, Input length, Channel] -> [Batch,Channel,Input length]
        seasonal_x = tf.transpose(seasonal_x, perm=[0, 2, 1])
        
        seasonal_x = self.Linear_Seasonal_seqlen0(seasonal_x) #(Batch,1,seq_len*4)
        seasonal_x = self.Linear_Seasonal_seqlen1(seasonal_x) #(Batch,1,pred_len)
        
        trend_x = self.Linear_Trend_channel(trend_init) #(Batch,seq_len,1)
        trend_x = tf.transpose(trend_x, perm=[0, 2, 1]) #(Batch,1,seq_len)
        trend_x = self.Linear_Trend_seqlen(trend_x) #(Btach,1,pred_len)
        
        x = seasonal_x + trend_x #(Batch,1, pred_len)
        
        return tf.transpose(x, perm=[0, 2, 1])  # to [Batch, pred_len, 1]

class LTSF_NLinear(tf.keras.Model):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len, name="LTSF_NLinear", **kwargs):
        super(LTSF_NLinear, self).__init__(name=name, **kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = tf.keras.layers.Dense(self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def call(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :]
        x = x - seq_last
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.Linear(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
    

class Decompose_FF_Linear(tf.keras.Model):
    """
    Decomposition-Feed Forward-Liner,该模型用于DQN网络中,基础模型,预测股市买卖动作;
    """

    def __init__(self, seq_len, in_features, out_features,
                 kernel_size=25, dropout=0.3, name="Decompose_FF_Linear", **kwargs):
        """
        seq_len: 输入序列长度;
        in_features: 输入预测特征数;
        pred_len: 输出序列长度
        out_features: 输出序列的特征数;
        kernel_size: moving_avg(即:AveragePooling1D)时的pool_size(窗口大小)
        """
        super(Decompose_FF_Linear, self).__init__(name=name, **kwargs)
        self.seq_len = seq_len
        # self.pred_len = pred_len
        self.in_features = in_features
        self.out_features = out_features

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
        # Feed Forward
        FF_hidden = 4 * in_features
        self.FF_Seasonal_Dense0 = tf.keras.layers.Dense(FF_hidden, activation='relu')
        self.FF_Seasonal_Dense1 = tf.keras.layers.Dense(in_features,activation='relu')
        self.FF_dropout = tf.keras.layers.Dropout(dropout)
        # Conv1D:
        self.Conv1D_Seasonal = tf.keras.layers.Conv1D(
            filters=out_features, kernel_size=seq_len, strides=1, padding='valid', activation=None)
        self.Conv1D_Trend = tf.keras.layers.Conv1D(
            filters=out_features, kernel_size=seq_len, strides=1, padding='valid', activation=None)

    def call(self, x):
        # x: [Batch, seq_len, in_features]
        seasonal_init, trend_init = self.decompsition(x)  # (Batch,seq_len,in_features)

        # Feed Forward:
        seasonal_x = self.FF_Seasonal_Dense0(
            seasonal_init)  # (Batch,seq_len,4*in_features)
        seasonal_x = self.FF_dropout(seasonal_x)
        seasonal_x = self.FF_Seasonal_Dense1(seasonal_x)  # (Batch,seq_len,in_features)
        seasonal_x = self.FF_dropout(seasonal_x)
        seasonal_x += seasonal_init

        # Conv1D:
        seasonal_x = self.Conv1D_Seasonal(seasonal_x)  # (Batch,1,out_features)
        trend_x = self.Conv1D_Trend(trend_init)  # (Batch,1,out_features)

        # 合并:
        x = seasonal_x + trend_x  # (Batch,1,out_features)

        return x






