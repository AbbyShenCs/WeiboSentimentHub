from snownlp import SnowNLP
from jieba import lcut

class SnowNLPQingGangFenXi:
    def __init__(self, *args, **kwargs):
        pass
   
    
    # 情感分析
    def feel(self,word):
        try:
            #python 舆情分析 nlp主题分析 （2）-结合snownlp与jieba库，提高分词与情感判断
            #因为舆情分析包含了情感分析我们为了区分两者的区别在舆情模块中包含了中文分词jiba功能
            wordList = lcut(word)
            data = SnowNLP(str(wordList)).sentiments
            print("情感评分（0.8以上为积极，0.1一下为负面")
            return data
        except Exception as err:
            return 0.5

    
    # 舆情分析
    def feel_super(self,word):       
        try:
            #单独的使用snow就行因为舆情分析包含了情感分析
            data = SnowNLP(word).sentiments
            print("情感评分（0.8以上为积极，0.1一下为负面")
            return data
        except Exception as err:
            return 0.5


    # 影响分析
    '''
    影响分析我们根据数据库中的所有数据来分析,
    主要做两部获取数据库中最大的热度标题,
    jiba分词获取出现频率最大的热搜，和最小的热搜
    '''
    def effect(self,word):       
        import collections # 词频统计库
        import imageio

        # 使用分词
        wordList = lcut(word)
        content = " ".join(wordList)  # 列表转换为空格隔开的字符串
        word_counts = collections.Counter(wordList) # 对分词做词频统计,词出现的次数

         # 获取前10000最高频的词
        word_counts_top10000 = word_counts.most_common(10000)
        print (type(word_counts_top10000[0]))
        print (word_counts_top10000[0][0])
        print (word_counts_top10000[0][1])

        print (word_counts_top10000[len(word_counts_top10000)-1][0])
        print (word_counts_top10000[len(word_counts_top10000)-1][1])


        effect_string  = "热搜中出现词最高的是："+word_counts_top10000[0][0]+ "\r\n"
        effect_string += "热搜中出现词最低的是："+word_counts_top10000[len(word_counts_top10000)-1][0] + "\r\n"
        # 输出检查
        #print (word_counts_top10000)
        return effect_string


    def stringToWordcloud(self,string,keyword):
        import collections # 词频统计库

        # 分词
        wordList = lcut(string)
        content = " ".join(wordList)  # 列表转换为空格隔开的字符串
        word_counts = collections.Counter(wordList) # 对分词做词频统计，词出现的次数

        #获取前10000最高频的词
        word_counts_top10000 = word_counts.most_common(10000)
        print (word_counts_top10000) # 输出检查

        #json 拼接字符串处理
        jsons = "["
        for k,v in word_counts_top10000:
            jsons+="{name:'"+k+"',"
            jsons+="value:"+str(v)+"},"
        jsons = jsons[0:len(jsons)-1]
        jsons += "]"
        return jsons
        i = 0
      