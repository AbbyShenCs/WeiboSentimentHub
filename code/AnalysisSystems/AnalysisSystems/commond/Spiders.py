import re
import requests
import pandas as pd
import datetime
import decimal
from AnalysisSystems.commond.Services import Services

class Spiders:
    def __init__(self, *args, **kwargs):
        pass


    def start_spiders(self):
        print("*"*100);
        print("爬虫开始");
        try:
            #第一步构造url还需要headrs
            url = 'https://tophub.today/n/KqndgxeLl9' 
            headers = {'user-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36 Edg/99.0.1150.39"}
            #第二步调用python 的requests
            #response=requests.get(url,headers=headers,timeout=3) 
            #response = requests.get(url, timeout=(8,15),headers = headers) 
            response = requests.get(url,headers = headers) 

            #第三步获取网站返回的信息，并且爬取最新的30个内容
            html = response.text
            #第四步使用python 的正则表达库获取titile标题和heat热度
            #正则学习地址
            #https://www.runoob.com/regexp/regexp-metachar.html
            titles = re.findall('<a href=".*?">.*?(.*?)</a>',html)[3:33]
            heat = re.findall('<td>(.*?)</td>',html)[:30]   

            #第五步获取爬取时间
            now_time = datetime.datetime.now()
            print(str(now_time).split(".")[0])

            #第六步创建空列表，把爬取到的热搜数据追加数据到空列表中
            data=[] 
            for i in range(30):
                #追加数据到空列表中
                if "万" in heat[i]:
                    #data.append([i+1,titles[i],heat[i][:]])
                    heatNumber = heat[i][:].replace("万","")
                    data.append([i+1,titles[i],heatNumber,str(now_time).split(".")[0]])

            #查看是否爬取到数据
            file=pd.DataFrame(data,columns=['排名','热搜事件','热度(万)','爬取时间'])
            print(file)

            i = 0
            #保存文件
            #file.to_excel('D:\\bbc\\微博热搜榜.xlsx')
            print("*"*100)
            print("爬虫结束")

            #第七步把获取的爬虫数据存到数据中，使用model层和数据库操作就行
            services =  Services()
            services.insert_sql(data)
        except  Exception as err:
            print("爬虫速度过快，稍后再试试")        
