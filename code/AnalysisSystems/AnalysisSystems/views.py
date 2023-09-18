"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template,redirect,url_for,request, make_response,Response,session
from AnalysisSystems import app
from AnalysisSystems.commond.Spiders import Spiders
from AnalysisSystems.commond.Services import Services
from AnalysisSystems.commond.SnowNLPQingGangFenXi import SnowNLPQingGangFenXi
import decimal

services =  Services()
spiders  =  Spiders()
snowNLPQingGangFenXi = SnowNLPQingGangFenXi()


#登录页面
@app.route('/')
@app.route('/login',methods=['GET', 'POST'])
def login():
    #获取http提交方式
    if request.method == 'GET':
        #直接返回视图的界面
        return render_template(
            'login.html',
        )
    else:
        #获取前端name中的数据
        username =  request.form.get('username')
        pwd     =  request.form.get('pwd')

        #将用户请求转发给相应的Model
        data = services.select_sql_login(username,pwd)
        if len(data) <= 0:
            return redirect("/login")

        #session 存储用户登录信息
        session['logged_in'] = username
        return redirect("/home")


#首页
@app.route('/home')
def home():
    """Renders the home page."""
    #session 检测之前login存储用户登录信息
    if (session['logged_in'] == None):
        return redirect("/login")

    #上一页下一页
    pre_page  = 0
    next_page = 6

    #获取请求是上一页还是下一页
    type_page = request.args.get("typePage");
    current_page = request.args.get("currentPage"); 
    '''
        分页机制
       第一页 0: limit 0 ,6
       第二页 1: limit 6 ,6
       第三页 2: limit 12,6
    '''
    if type_page !=None:
        #处理上一页
        if type_page  ==  "pre" and int(current_page) !=0:
            pre_page = int(current_page) - 6
            next_page = int(current_page)
        #处理上一页
        elif type_page == "next" :
            pre_page = int(current_page)
            next_page = int(current_page) + int(current_page)
    else:
        current_page = 0

    #将用户请求转发给相应的Model
    data =  services.select_sql(int(current_page), 6)


    #可视化板块
    #获取最新20个数据，统计并在页面上显示
    new_title = []
    new_title_level = []
    level = 1
    new_title_hot = []
    #调用Model获取最新的20个数据
    fist20 = services.select_sql_fist20()

    #代迭20个数据，并把它分别以split函数处理完添加到相应的集合中     new_title = []， new_title_level = []，new_title_hot = []
    for x in fist20:
        new_title.append(x.split(" ")[1]);
        new_title_level.append(level);
        new_title_hot.append(x.split(" ")[2]);
        level = level+1

    #所有事情处理完以后把结果信息返回给index.html界面中
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
        datas = data,
        pre   = pre_page,
        next  = next_page,
        new_title = new_title,
        new_title_level = new_title_level,
        new_title_hot       = new_title_hot
    )


#爬虫
@app.route('/StartSpiders')
def StartSpiders():
    #session 检测之前login存储用户登录信息
    if (session['logged_in'] == None):
        return redirect("/login")

    #启动爬虫
    spiders.start_spiders()
    response = Response()
    return response


#可视化类页面
@app.route('/echarts')
def echarts():
    """Renders the contact page."""
    #session 检测之前login存储用户登录信息
    if (session['logged_in'] == None):
        return redirect("/login")

    #通过前端获取的类型画出不同的图形
    parm = request.args.get("type");
    #前端用于区分画出那个图形
    echartType = ""

    #如果是折线图
    if parm == None or parm == "":
        echartType = "line"

        #获取最新20个数据，统计并在页面上显示
        #新闻标题
        new_title = []
        new_title_level = []
        level = 1
        #新闻热搜
        new_title_hot = []

        #将用户请求转发给相应的Model，从数据库中获取前20个数据
        fist20 = services.select_sql_fist20()
        #代迭20个数据，并把它分别以split函数处理完添加到相应的集合中     new_title = []， new_title_level = []，new_title_hot = []
        for x in fist20:
            new_title.append(x.split(" ")[1]);
            new_title_level.append(level);
            new_title_hot.append(x.split(" ")[2]);
            level = level+1

        #最后返回数据给前端
        return render_template(
        'echarts.html',
        title  = 'echarts',
        year   = datetime.now().year,
        type   = echartType,
        new_title = new_title,
        new_title_hot       = new_title_hot
        )    
    #如果是云图
    else:      
        echartType = "word"
        #将用户请求转发给相应的Model，从数据库中获取所有数据统计并在页面上显示云词
        new_title_all = ""
        all = services.select_sql_all()
        for x in all:
            #把所有人热搜词，标题等组合到一个字符串变量中，最后通过分词算法
            new_title_all += x.split(" ")[1];
        
        #调用SnowNLPQingGangFenXi类中的stringToWordcloud获取云词数据
        snowNLPQingGangFenXi =  SnowNLPQingGangFenXi()
        data =  snowNLPQingGangFenXi.stringToWordcloud(new_title_all,"")

        #最后返回数据给前端
        return render_template(
        'echarts.html',
        title  = 'echarts',
        year   = datetime.now().year,
        type   = echartType,
        data   = data) 



#分析管理页面
@app.route('/analysis')
def analysis():
    """Renders the about page."""
    #session 检测之前login存储用户登录信息
    if (session['logged_in'] == None):
        return redirect("/login")

    #通过前端发送的数据，获取的类型然后分析步同的结果
    analysis_type = request.args.get("analysisType")
    #分析的结果
    messages = "";
    #获取所有数据
    new_title_all = ""
    #最小最大热度，最小的热度不能是0 所以需要默认定义一个值
    new_min_hot = 50
    new_max_hot = 0

    #将用户请求转发给相应的Model，从数据库中获取所有热搜信息
    all = services.select_sql_all_row()
    for x in all:
        #把所有人热搜词，标题等组合到一个字符串变量中，最后通过分词算法
        new_title_all += x[1];
        temp = float(str(decimal.Decimal(x[2]).quantize(decimal.Decimal('0.00'))))
        #这里是查询最大和最小热搜度
        if new_min_hot > temp:
            new_min_hot = temp
        if new_max_hot < temp:
            new_max_hot = temp

    #情感分析       
    if analysis_type   == 'feel':
        #将用户请求转发给相应的Model，从数据库中获取前20个数据
        all = services.select_sql_fist20()
        for x in all:
            new_title_all = x.split(" ")[1]
            #拼接分析的结果到messages
            messages += "热搜标题："+new_title_all+"\r\n" 
            messages += "现热搜中情感分析结果是：\r\n" 
            messages += str(snowNLPQingGangFenXi.feel(new_title_all))
            messages += "\r\n"
            messages += "情感评分（0.6以上为积极，0.2一下为负面\r\n\r\n"
    #舆情分析
    elif analysis_type == 'feel_super':
        #将用户请求转发给相应的Model，从数据库中获取前20个数据
        all = services.select_sql_fist20()
        for x in all:
            new_title_all = x.split(" ")[1]
            #拼接分析的结果到messages
            messages += "热搜标题："+new_title_all+"\r\n" 
            messages += "现热搜中舆情分析结果是：\r\n" 
            messages += str(snowNLPQingGangFenXi.feel_super(new_title_all))
            messages += "\r\n"
            messages += "情感评分（0.6以上为积极，0.2一下为负面\r\n\r\n"
    #影响分析
    elif analysis_type == 'effect':
        #拼接分析的结果到messages
        messages = snowNLPQingGangFenXi.effect(new_title_all)
        messages += "现热搜中热度最高的是：" + services.select_sql_max_hot(new_max_hot)[0][1]+ "\r\n" 
        messages += "热度："+ str(new_max_hot)+ "万\r\n"
        messages += "现热搜中热度最低的是：" + services.select_sql_min_hot(new_min_hot)[0][1]+"\r\n" 
        messages += "热度："+ str(new_min_hot)+ "万\r\n"

    #当处理完不同分支语句以后返回结果信息到界面显示
    return render_template(
        'analysis.html',
        title='Analysis',
        year=datetime.now().year,
        messages=messages)



#密码重置管理页面
@app.route('/pwdchange',methods=['GET', 'POST'])
def pwdchange():
    """Renders the about page."""
    #session 检测之前login存储用户登录信息
    if (session['logged_in'] == None):
        return redirect("/login")

    #获取http提交方式
    if request.method == 'GET':
        #直接返回视图的界面
        return render_template(
            'pwdchange.html',
        )
    else:
        newpwd     =  request.form.get('newpwd')
        confirmpwd =  request.form.get('confirmpwd')

        services.changepwd_sql_login(newpwd,confirmpwd)

        #当处理完不同分支语句以后返回结果信息到界面显示
        session['logged_in'] = None
        return redirect("/login")