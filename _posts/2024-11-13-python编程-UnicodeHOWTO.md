---
layout: post
category: Python编程
---

# Unicode HOWTO
## Unicode 简介
字符 character

编码点 code point：字符在字符集中的序号，非负整数

字形 glyph：字符的显示形式，跟渲染方式有关

编码 encoding：字符在内存中的二进制表示形式，由一个或多个编码单元 code units 组成

编码单元 code units：对应一个或多个字节

中文汉字范围，即中日韩统一表意文字**，**编码范围为U+4E00–U+9FFF。

## Python的Unicode支持
### String 类
支持使用ASCII和转义字符表示Unicode字符

```python
"\N{GREEK CAPITAL LETTER DELTA}" 
```
上述代码打印出希腊字母delta（根据系统不同，可能打印出'\\u0394'）
编码点和字符的转换：`chr()`和`ord()`能够完成单字符和编码点（整数）间的转换。例如：

```python
>>> ord('中')  # 求汉字'中'的编码点
20013
>>> chr(20013)  # 将编码点转换为字符
'中'
>>> hex(ord('中'))  # 以十六进制形式表示
'0x4e2d'
>>> chr(0x4e2d)  # 将编码点转换为字符
'中'
>>> '\u4e2d'  # 字符'中'的编码点书写形式
'中'
```
### 转换成Bytes
`bytes.decode()`和`str.encode()`完成`str`和`bytes`类型之间的转换。

### 源码中的Unicode 字面量
字符既可以用字面形式书写，如：'a'，也可以用编码点形式书写，如:'\\x61'。对于不可打印字符，只能用编码点形式书写。

Unicode字面量书写形式：

* \\x，后面跟2个16进制整数，表示1个Unicode字符
* \\u，后面跟4个16进制整数
* \\U，后面跟8个16进制整数

```python
'\x61'
'\u0061'
'\U00000061'
'a'
```
上述均表示字符'a'。

源码默认UTF-8编码，可通过注释显示声明。

```python
#!/usr/bin/env python
# -*- coding: latin-1 -*-
u = 'abcdé'
print(ord(u[-1]))
```
`-*-`表示该注释是特殊的（Emacs风格），`coding=name`亦可。

### 读写Unicode数据
`open()`的`error`参数可以设置编码错误的处理方法。

### 编写支持Unicode程序的建议
> Software should only work with Unicode strings internally, decoding the input data as soon as possible and encoding the output only at the end.

### 文件编码转换
`StreamRecoder`类能够转换数据流的编码

```python
new_f = codecs.StreamRecoder(f,
    # en/decoder: used by read() to encode its results and
    # by write() to decode its input.
    codecs.getencoder('utf-8'), codecs.getdecoder('utf-8'),
    # reader/writer: used to read and write to the stream.
    codecs.getreader('latin-1'), codecs.getwriter('latin-1') )
```
上述代码，数据写入时，通过decoder按照UTF-8解码成中间形式，然后按照latin-1编码，写到下游；

数据读取时，通过reader按照latin-1解码成中间形式，然后按照UTF-8编码，返回。

### 未知编码的文件
使用errors参数，指定相应的编码错误处理器。

```python

with open(fname, 'r', encoding="ascii", errors="surrogateescape") as f:
    data = f.read()

# make changes to the string 'data'

with open(fname + '.new', 'w',
          encoding="ascii", errors="surrogateescape") as f:
    f.write(data)
```
上述代码只修改ASCII字符，其他字符保持原样。

## str和bytes类
简单将`str` 对象理解为unicode字符的码点**序列**，每个元素对应一个字符，对应的二进制值是码点（ordinal）；

将`bytes` 对象理解为编码后的字节**序列**，每个元素是一个0-255的整数，1个或多个元素对应一个字符，且二进制值和编码方式有关，不能直接对应unicode码点。

### str
> Strings are immutable sequences of Unicode code points.

`str`类是**不可变**的Unicode编码点的序列。
`str.encode` 方法将str对象转换成`bytes`对象。

### bytes
> Bytes objects are immutable sequences of single bytes.

`bytes`类是**不可变**的字节序列。

`bytes`类是内置类型，`bytes`对象的创建，可以通过：

* 直接字面量创建：`b' Only ASCII character are permmitted in bytes literals'` ，只有ascii字符可以直接写成bytes字面量，超过127的要用转义字符，b'\\xe4\\xbd\\xa0'
* 使用十六进制数字创建，空格被忽略：

```python
>>> bytes.fromhex('2Ef0 F1f2  ')
# 2E是'.'的ascii编码，直接以'.'显示
b'.\xf0\xf1\xf2'
```
`bytes.decode` 方法将bytes对象转换成`str`对象。

每个元素都是整数 int：

```python
# print 49~53
for i in b'12345':
    print(i)

a = b'1'
print(type(a[0]))  # print int
```
### bytearray
bytearray 对象相当于可变的 bytes 对象。

*class bytearray(\[\_source\_\[, \_encoding\_\[, \_errors\_\]\]\])*

通过构造函数创建bytearray对象：

* 空实例：`bytearray()`
* 可以指定容量：`bytearray(10)`
* 从列表或可迭代对象创建：`bytearray(range(20))`
* 从bytes对象复制：`bytearray(b'Hi')`

bytes/bytearray除了可变性差异外，几乎可以认为等同。

### 类型转换
| |str|bytes|int|
| ----- | ----- | ----- | ----- |
|str| |str.encode|int()|
|bytes|bytes.decode| |int.from\_bytes|
|int|str()|int.to\_bytes| |

更多类型转换可以使用`struct` 模块。