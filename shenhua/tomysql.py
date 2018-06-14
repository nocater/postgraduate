import pandas as pd
from sqlalchemy import create_engine
import mysql.connector

PATH = r'C:\Users\chenshuai\Documents\神华黄骅\文档\\'

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

# planload data
# planload = pd.read_excel(PATH+'planload.xlsx')
# planload.rename(columns={'   ':'id'}, inplace=True)

# create_engine("数据库类型+数据库驱动://数据库用户名:数据库密码@IP地址:端口/数据库"，其他参数)
engine=create_engine("mysql+pymysql://root:2wsx3edc@localhost:3306/shenhua?charset=utf8", echo=True)
workflow.to_sql('workflow', con=engine, index=False)
planload.to_sql('planload', con=engine, index=False)
