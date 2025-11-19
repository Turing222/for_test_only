import time
from datetime import datetime
b=datetime.now()
c=datetime.now().timestamp()

#print(datetime.strptime(c,'%Y-%m-%d %H:%M:%S'))

# sort a list of numbers
d=[5,2,9,1,5,6]
print(*d)
print(d)
dict_1={"a":"1","b":"2"}
dict_2={"c":"1","d":"2"}
a=2
b=3

dict_3={**dict_1,**dict_2}
print(dict_3)
print("a:",a)
#print(**dict_1,a,b)

def process_options(aa, bb, **kwargs):
    print(f"位置参数 a: {aa}")
    print(f"位置参数 b: {bb}")
    print(f"收集到的其他关键字参数 kwargs: {kwargs}")

process_options(1,2,**dict_3)



