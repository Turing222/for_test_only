#内置函数

#十六进制表示 0x为前缀
a=0x1A
print(a)
#使用logging代替print
import logging
a=0x1A

mylogger = logging.getLogger('my_logger')
mylogger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

mylogger.addHandler(console_handler)


#console_handler = logging.StreamHandler()
#console_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(
#    level=logging.INFO,
#    format='%(asctime)s - %(levelname)s - %(message)s'
#)
#logging.info("my a:%s",a)
mylogger.propagate
print(mylogger.propagate)
mylogger.info("mylogger massage: a:%s",a)
#Unicode编码