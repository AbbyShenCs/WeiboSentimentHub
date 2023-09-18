import decimal
from AnalysisSystems.commond.MySqlHelper import MySqlHelper


#它的作用和mvc model一样，使用MySqlHelper类中的query方法，
#把sql语句写好传给它就可以和数据库通信了，最后把获取的结果返回给view.py 使用
class Services:
    def __init__(self, *args, **kwargs):
        pass

    #获取的爬虫数据存到数据中
    def insert_sql(self,data_list):
        mysql = MySqlHelper()

        for data in data_list:  
            id  = data[0]
            titile = data[1]
            heat   = data[2]
            hotTime = data[3]
            #调用MySqlHelper的query，这里先查询数据库中是否有重复的热搜如果没有插入，如果有插入重复的数据
            reuslt =  mysql.query("select * from hotseacher where Titile='"+str(titile)+"';","")
            #如果没有插入
            if reuslt == 0:                
                mysql.query("insert into hotseacher (`Titile`,`Heat`,`HotTimes`) values('"+titile+"', '"+heat+"', '"+hotTime+"');", "")
                #提交插入数据
                mysql.connent.commit()
        mysql.end()        


    '''
       第一页 0: limit 0 ,6
       第二页 1: limit 6 ,6
       第三页 2: limit 12,6
    '''
    #上一页和下一页的sql操作
    def select_sql(self,current_page, sums):

        mysql = MySqlHelper()
        #通过sql语句获取前6条数据，然后根据用户提交的当前页面书current_page获取下一页或是上一页的数据
        reuslt =  mysql.query("select * from hotseacher  order by Id desc limit "+str(current_page)+" , 6", "")
        data = []
        #如果上一页或是下一页的数据有数据，把它追加到data集合中并返回
        if reuslt > 0:
            pass
            for row in mysql.cursor.fetchall():
                print(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.0'))) + " " + str(row[3]))
                #把它追加到data集合中并返回
                data.append(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.0'))) + " " + str(row[3]));
            mysql.end()
        #如果上一页或是下一页的数据有数据，把它追加到data集合中并返回
        return data


    #获取前20条数据
    def select_sql_fist20(self):
        #通过sql语句获取前20条数据
        mysql = MySqlHelper()
        reuslt =  mysql.query("select * from hotseacher  order by Id desc limit 0 , 20", "")
        data = []

        #如果有数据，把它追加到data集合中并返回
        if reuslt > 0:
            pass
            for row in mysql.cursor.fetchall():
                print(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.0'))) + " " + str(row[3]))
                #把它追加之前我们需要先把他字符处理以后，在把它到data集合中并返回
                data.append(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.0'))) + " " + str(row[3]));
            mysql.end()
        return data

    
    #获取所有数据
    def select_sql_all(self):
        mysql = MySqlHelper()
        #通过sql语句获取所有数据
        reuslt =  mysql.query("select * from hotseacher", "")
        data = []

        #如果有数据，把它追加到data集合中并返回
        if reuslt > 0:
            pass
            for row in mysql.cursor.fetchall():
                print(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.00'))) + " " + str(row[3]))                
                #把它追加之前我们需要先把他字符处理以后，在把它到data集合中并返回
                data.append(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.00'))) + " " + str(row[3]));
            mysql.end()
        return data


    #获取所有数据
    def select_sql_all_row(self):
        mysql = MySqlHelper()
        #通过sql语句获取所有数据
        reuslt =  mysql.query("select * from hotseacher", "")
        data = []
        if reuslt > 0:
            pass
            for row in mysql.cursor.fetchall():
                #print(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.0'))) + " " + str(row[3]))
                #把它追加到data集合中并返回
                data.append(row);
            mysql.end()
        return data


    #获取最小热度数据
    def select_sql_min_hot(self,min):
        mysql = MySqlHelper()
        #通过sql语句获取最小热度数据
        reuslt =  mysql.query("select * from hotseacher where Heat="+str(min), "")
        data = []
        
        #如果有数据，把它追加到data集合中并返回
        if reuslt > 0:
            pass
            for row in mysql.cursor.fetchall():
                #print(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.0'))) + " " + str(row[3]))
                #把它追加到data集合中并返回
                data.append(row);
            mysql.end()
        return data


    #获取最大热度数据
    def select_sql_max_hot(self,max):
        mysql = MySqlHelper()
        #通过sql语句获取最大热度数据
        reuslt =  mysql.query("select * from hotseacher where Heat="+str(max), "")
        data = []
        if reuslt > 0:
            pass
            for row in mysql.cursor.fetchall():
                #print(str(row[0])+" "+ row[1] + " "+ str(decimal.Decimal(row[2]).quantize(decimal.Decimal('0.0'))) + " " + str(row[3]))
                #把它追加到data集合中并返回
                data.append(row);
            mysql.end()
        return data


    #登录sql操作
    def select_sql_login(self,username,password):
        mysql = MySqlHelper()
        #通过sql语句获取用户信息
        reuslt =  mysql.query("select * from admin where UserName='"+username+"' and PWD='"+password+"'", "")
        data = []
        
        #如果有用户的数据，把它追加到data集合中并返回，登录成功，如果没有登录失败
        if reuslt > 0:
            pass
            for row in mysql.cursor.fetchall():
                #print(str(row[0])+" "+ row[1] + " ")
                #把它追加之前我们需要先把他字符处理以后，在把它到data集合中并返回
                data.append(str(row[0])+" "+ row[1] + " ");
            mysql.end()
        return data


    #密码重置sql操作
    def changepwd_sql_login(self,username,password):
        mysql = MySqlHelper()
        #通过sql语句获取用户信息
        reuslt =  mysql.query("update admin set PWD='"+password+"' where Id=1", "")
        mysql.connent.commit()
        return reuslt