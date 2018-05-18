import pandas as pd
import numpy as np

BASE_PATH = r'C:\Users\chenshuai\Documents\airquality'


def getY_2014():
    """
    获取Y 浓度数据
    :return:
    """
    pd_pm = pd.read_excel(BASE_PATH+r'\数据\2014-2016逐日浓度数据.xls', header=1)
    pd_pm = pd_pm.rename(columns={'日期':'date','石家庄市':'sjz','邢台市':'xt'})
    bounds = [1085, 2174, 3263, 4352, 5441] #1086 +4
    pd_pm2 = pd_pm.iloc[:bounds[0], :]
    pd_pm10 = pd_pm.iloc[bounds[0]+4:bounds[1], :]
    pd_so2 = pd_pm.iloc[bounds[1]+4:bounds[2], :]
    pd_no2 = pd_pm.iloc[bounds[2]+4:bounds[3], :]
    pd_co = pd_pm.iloc[bounds[3]+4:bounds[4], :]
    pd_o3 = pd_pm.iloc[bounds[4]+4:-1, :]

    # print(pd_pm2.shape,  pd_pm2.iloc[0,0], pd_pm2.iloc[-1,0])
    # print(pd_pm10.shape, pd_pm10.iloc[0,0], pd_pm10.iloc[-1,0])
    # print(pd_so2.shape, pd_so2.iloc[0,0], pd_so2.iloc[-1,0])
    # print(pd_no2.shape, pd_no2.iloc[0,0], pd_no2.iloc[-1,0])
    # print(pd_co.shape, pd_co.iloc[0,0], pd_co.iloc[-1,0])
    # print(pd_o3.shape, pd_o3.iloc[0,0], pd_o3.iloc[-1,0])
    # print(pd_pm10.columns)

    # 更改列名
    pd_pm2 = pd_pm2.rename(columns={'sjz':'sjz_pm2','xt':'xt_pm2'})
    pd_pm2 = pd_pm2.reset_index(drop=True)
    pd_pm10 = pd_pm10.rename(columns={'sjz':'sjz_pm10','xt':'xt_pm10'})
    pd_pm10 = pd_pm10.reset_index(drop=True)
    pd_so2 = pd_so2.rename(columns={'sjz':'sjz_so2','xt':'xt_so2'})
    pd_so2 = pd_so2.reset_index(drop=True)
    pd_no2 = pd_no2.rename(columns={'sjz':'sjz_no2','xt':'xt_no2'})
    pd_no2 = pd_no2.reset_index(drop=True)
    pd_co = pd_co.rename(columns={'sjz':'sjz_co2','xt':'xt_co2'})
    pd_co = pd_co.reset_index(drop=True)
    pd_o3 = pd_o3.rename(columns={'sjz':'sjz_o3','xt':'xt_o3'})
    pd_o3 = pd_o3.reset_index(drop=True)

    # 合并到一个df_y
    df_y = pd.concat([pd_pm2, pd_pm10.iloc[:, 1:], pd_so2.iloc[:, 1:],
                      pd_no2.iloc[:, 1:], pd_co.iloc[:, 1:], pd_o3.iloc[:, 1:]], axis=1)


    # print(df_y.shape, pd_pm2.shape, df_y.columns)
    # 转换成float
    cols = df_y.columns[1:]
    df_y[cols] = df_y[cols].apply(pd.to_numeric, errors='coerce')

    #转换日期
    df_y.date = pd.to_datetime(df_y.date)
    # 只保留年月日
    df_y.date = df_y.date.map(lambda x: x.strftime('%Y-%m-%d'))
    return df_y


def getY_2017():
    """
    获取Y 浓度数据
    :return:
    """
    pd_pm = pd.read_excel(BASE_PATH+r'\数据\2017-1-2-2-28逐日浓度数据.xls', header=1)
    pd_pm = pd_pm.rename(columns={'日期':'date','石家庄市':'sjz','邢台市':'xt'})
    bounds = [58, 120, 182, 244, 306] #1086 +4
    pd_pm2 = pd_pm.iloc[:bounds[0], :]
    pd_pm10 = pd_pm.iloc[bounds[0]+4:bounds[1], :]
    pd_so2 = pd_pm.iloc[bounds[1]+4:bounds[2], :]
    pd_no2 = pd_pm.iloc[bounds[2]+4:bounds[3], :]
    pd_co = pd_pm.iloc[bounds[3]+4:bounds[4], :]
    pd_o3 = pd_pm.iloc[bounds[4]+4:-1, :]

    print(pd_pm2.shape,  pd_pm2.iloc[0,0], pd_pm2.iloc[-1,0])
    print(pd_pm10.shape, pd_pm10.iloc[0,0], pd_pm10.iloc[-1,0])
    print(pd_so2.shape, pd_so2.iloc[0,0], pd_so2.iloc[-1,0])
    print(pd_no2.shape, pd_no2.iloc[0,0], pd_no2.iloc[-1,0])
    print(pd_co.shape, pd_co.iloc[0,0], pd_co.iloc[-1,0])
    print(pd_o3.shape, pd_o3.iloc[0,0], pd_o3.iloc[-1,0])
    print(pd_pm10.columns)

    # 更改列名
    pd_pm2 = pd_pm2.rename(columns={'sjz':'sjz_pm2','xt':'xt_pm2'})
    pd_pm2 = pd_pm2.reset_index(drop=True)
    pd_pm10 = pd_pm10.rename(columns={'sjz':'sjz_pm10','xt':'xt_pm10'})
    pd_pm10 = pd_pm10.reset_index(drop=True)
    pd_so2 = pd_so2.rename(columns={'sjz':'sjz_so2','xt':'xt_so2'})
    pd_so2 = pd_so2.reset_index(drop=True)
    pd_no2 = pd_no2.rename(columns={'sjz':'sjz_no2','xt':'xt_no2'})
    pd_no2 = pd_no2.reset_index(drop=True)
    pd_co = pd_co.rename(columns={'sjz':'sjz_co2','xt':'xt_co2'})
    pd_co = pd_co.reset_index(drop=True)
    pd_o3 = pd_o3.rename(columns={'sjz':'sjz_o3','xt':'xt_o3'})
    pd_o3 = pd_o3.reset_index(drop=True)

    # 合并到一个df_y
    df_y = pd.concat([pd_pm2, pd_pm10.iloc[:, 1:], pd_so2.iloc[:, 1:],
                      pd_no2.iloc[:, 1:], pd_co.iloc[:, 1:], pd_o3.iloc[:, 1:]], axis=1)

    # print(df_y.shape, pd_pm2.shape, df_y.columns)
    # 转换成float
    cols = df_y.columns[1:]
    df_y[cols] = df_y[cols].apply(pd.to_numeric, errors='coerce')

    #转换日期
    df_y.date = pd.to_datetime(df_y.date)
    # 只保留年月日
    df_y.date = df_y.date.map(lambda x: x.strftime('%Y-%m-%d'))
    df_y.info()
    return df_y


def getStaticStability_2014():
    df_staticstability = pd.read_excel(BASE_PATH+r'\数据\2014-2016年石家庄和邢台静稳指数一天四次.xls')
    df_staticstability = df_staticstability[['站名', '日期', '静稳指数']]
    df_staticstability.columns = ['name', 'date', 'staticstability']

    # 转换日期
    df_staticstability.date = pd.to_datetime(df_staticstability['date'])
    # 转换float
    df_staticstability.staticstability = pd.to_numeric(df_staticstability.staticstability)

    # print(df_staticstability.info())
    df_sjz = df_staticstability[df_staticstability.name == '石家庄']
    df_xt = df_staticstability[df_staticstability.name == '邢台']
    # 只保留年月日
    df_sjz.date = df_sjz.date.map(lambda x: x.strftime('%Y-%m-%d'))
    df_xt.date = df_xt.date.map(lambda x: x.strftime('%Y-%m-%d'))

    # 分组求均值
    df_mean_sjz = df_sjz.groupby('date')['staticstability'].mean()
    # S转为DF 将index成列
    df_mean_sjz = df_mean_sjz.to_frame()
    df_mean_sjz = df_mean_sjz.reset_index()
    df_mean_sjz.reset_index(drop=True)

    df_mean_xt = df_xt.groupby('date')['staticstability'].mean()
    df_mean_xt = df_mean_xt.to_frame()
    df_mean_xt = df_mean_xt.reset_index()
    df_mean_xt.reset_index(drop=True)

    # print(df_staticstability)
    df_sss = pd.concat([df_mean_sjz, df_mean_xt.loc[:, 'staticstability']], axis=1)
    df_sss.columns = ['date', 'sjz_staticstability', 'xt_staticstability']
    return df_sss


def getStaticStability_2017():
    df_staticstability = pd.read_excel(BASE_PATH+r'\数据\2017-1-2-2-28静稳指数.xls')
    df_staticstability = df_staticstability[['站名', '日期', '静稳指数']]
    df_staticstability.columns = ['name', 'date', 'staticstability']

    # 转换日期
    df_staticstability.date = pd.to_datetime(df_staticstability['date'])
    # 转换float
    df_staticstability.staticstability = pd.to_numeric(df_staticstability.staticstability)

    # print(df_staticstability.info())
    df_sjz = df_staticstability[df_staticstability.name == '石家庄']
    df_xt = df_staticstability[df_staticstability.name == '邢台']
    # 只保留年月日
    df_sjz.date = df_sjz.date.map(lambda x: x.strftime('%Y-%m-%d'))
    df_xt.date = df_xt.date.map(lambda x: x.strftime('%Y-%m-%d'))

    # 分组求均值
    df_mean_sjz = df_sjz.groupby('date')['staticstability'].mean()
    # S转为DF 将index成列
    df_mean_sjz = df_mean_sjz.to_frame()
    df_mean_sjz = df_mean_sjz.reset_index()
    df_mean_sjz.reset_index(drop=True)

    df_mean_xt = df_xt.groupby('date')['staticstability'].mean()
    df_mean_xt = df_mean_xt.to_frame()
    df_mean_xt = df_mean_xt.reset_index()
    df_mean_xt.reset_index(drop=True)

    # print(df_staticstability)
    df_sss = pd.concat([df_mean_sjz, df_mean_xt.loc[:, 'staticstability']], axis=1)
    df_sss.columns = ['date', 'sjz_staticstability', 'xt_staticstability']
    return df_sss


def getQing_2014():
    df_qing = pd.read_csv(BASE_PATH+r'\qing2014.csv', encoding='utf-8')
    # 转换日期
    df_qing.date = pd.to_datetime(df_qing['date'])
    # 只保留年月日
    df_qing.date = df_qing.date.map(lambda x: x.strftime('%Y-%m-%d'))

    df_qing.iloc[:, 1:] = df_qing.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    df_qing = df_qing.fillna(value=0)
    return df_qing


def getQing_2017():
    df_qing = pd.read_csv(BASE_PATH+r'\qing2017.csv', encoding='utf-8')
    # 转换日期
    df_qing.date = pd.to_datetime(df_qing['date'])
    # 只保留年月日
    df_qing.date = df_qing.date.map(lambda x: x.strftime('%Y-%m-%d'))

    df_qing.iloc[:, 1:] = df_qing.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    df_qing = df_qing.fillna(value=0)
    return df_qing

if __name__ == "__main__":
    # 污染数据
    df_y = getY_2014()
    # 静稳数据
    df_ss = getStaticStability_2014()
    # 合并
    df = pd.merge(df_y, df_ss, how='left', on='date')
    print(df.columns)
    # 2014-7-[3,5,7,8,9] 的静稳数据没有  用前后数据填充
    df.loc[(df.sjz_staticstability.isnull()), 'sjz_staticstability'] = np.linspace(10.87,10.44,7)[1:-1]
    df.loc[(df.xt_staticstability.isnull()), 'xt_staticstability'] = np.linspace(10.64,10.32,7)[1:-1]

    df_qing = getQing_2014()
    df = pd.merge(df, df_qing, how='left', on='date')
    # print(df.info(), df.shape)
    df_2014 = df

    df_2017 = getY_2017()
    df_ss_2017 = getStaticStability_2017()
    df_2017 = pd.merge(df_2017, df_ss_2017, how='left', on='date')
    df_qing_2017 = getQing_2017()
    df_2017 = pd.merge(df_2017, df_qing_2017, how='left', on='date')
    df = pd.concat([df_2014, df_2017])

    # #转换日期
    # df.date = pd.to_datetime(df.date)
    # # 只保留年月日
    # df.date = df.date.map(lambda x: x.strftime('%Y-%m-%d'))
    # 陈帅数据导出
    df.to_csv(BASE_PATH+r'\chen2014-2017.csv', index=False)