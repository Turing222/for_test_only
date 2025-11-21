#内置函数

#十六进制表示 0x为前缀
a=0x1A
print(a)
#使用logging代替print
import logging
a=0x1A

#自定义记录器
mylogger = logging.getLogger('my_logger')
mylogger.setLevel(logging.INFO)

#为记录器创建处理器 种类是输出控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

#输出对象为指定文件
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(mylogger.debug)

#格式化器
formatter = mylogger.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(message)s'
)

mylogger.addHandler(console_handler)

#根记录器
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

logging
mylogger.propagate

#自定义项目是否继承
print(mylogger.propagate)


mylogger.info("mylogger massage: a:%s",a)


#Unicode编码