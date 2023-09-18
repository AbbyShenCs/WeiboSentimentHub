#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import pymysql
from AnalysisSystems.commond.Config import *


#连接数据库使用的类，数据库连接类
class MySqlHelper(object):
    def __init__(self):
        #第一步先获取Config中获取数据库信息用户名密码配置信息等
        self.__connect_dict = connect_dict_windows_config 
        #self.__connect_dict = Config.connect_dict_linux_config 
        #第二步连接数据库
        self.connent = pymysql.Connect(**self.__connect_dict)
        #第三步获取连接的下标，
        self.cursor = self.connent.cursor()
    
    #第四步定义一个查询添加编辑删除的函数
    def query(self,sql,parms):
        return self.cursor.execute(sql)
        for row in self.cursor.fetchall():
            print("name:\t " , row)
        print(f"Sum row: {self.cursor.rowcount}")

    #第五步关闭连接和下标，
    def end(self):
        self.cursor.close()
        self.connent.close()