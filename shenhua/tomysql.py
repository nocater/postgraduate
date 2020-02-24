import pandas as pd
from sqlalchemy import create_engine
import mysql.connector
import re

PATH = r'C:\Users\chenshuai\Documents\神华黄骅\数据\\'

def analysisCD_S():
    # 列出工作流程定义中 翻车机和堆料机的对应
    cd_s = []
    df = pd.read_excel(PATH + '作业流程定义.xls')
    for cd in range(1, 14):
        for i in list(df.设备串):
            if re.match(r'CD\d+.*S[/R\d?|\d+]', i):
                cd_s.append(str(i.split('-')[0]+'-'+i.split('-')[-1]))
                # print(i, i.split('-')[-1])

    cd_s = sorted(list(set(cd_s)))
    print(cd_s)
    df = pd.DataFrame(cd_s)
    df.to_csv(PATH+'CD_S.csv')

def exceltomysql():
    # workflow data
    workflow = pd.read_excel(PATH+'workflow.xlsx')
    workflow.rename(columns={'   ':'ID'}, inplace=True)
    # chage the value type
    # workflow.FID = workflow.FID.astype('int64')
    # date
    # workflow.WORKDATE = pd.to_datetime(workflow.WORKDATE)
    for col in ['WORKDATE', 'STARTTIME', 'ENDTIME', 'MODIFYTIME', 'CONFIRMTIME']:
        workflow[col] = pd.to_datetime(workflow[col])

    print(workflow.columns)
    workflow.info()

    # create_engine("数据库类型+数据库驱动://数据库用户名:数据库密码@IP地址:端口/数据库"，其他参数)
    engine=create_engine("mysql+pymysql://root:2wsx3edc@localhost:3306/shenhua?charset=utf8", echo=True)
    workflow.to_sql('workflow', con=engine, index=False)

    # planload data
    # planload = pd.read_excel(PATH+'planload.xlsx')
    # planload.rename(columns={'   ':'id'}, inplace=True)

def shangqing():
    pass
    df = pd.read_excel(r'C:\Users\chenshuai\Desktop\acl_group.xlsx')
    print(df.columns)
    engine = create_engine("mysql+pymysql://root:2wsx3edc@localhost:3306/shenhua?charset=utf8", echo=True)
    df.to_sql('acl', con=engine, index=False)

if __name__ == '__main__':
    shangqing()