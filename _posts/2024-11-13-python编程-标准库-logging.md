---
layout: post
category: Python编程
---

# 标准库: logging
## Logging HOWTO
日志的构成：时间、文件名、函数名、等级、行数、进程ID、消息等等。

### When to use logging

|Task you want to perform|The best tool for the task|
| :----- | :----- |
|Display console output for ordinary usage of a command line script or program|print()|
|Report events that occur during normal operation of a program (e.g. for status monitoring or fault investigation)|logging.info() (or logging.debug() for very detailed output for diagnostic purposes)|
|Issue a warning regarding a particular runtime event|**warnings.warn()** in library code if the issue is avoidable and the client application should be modified to eliminate the warning. **logging.warning()** if there is nothing the client application can do about the situation, but the event should still be noted|
|Report an error regarding a particular runtime event|Raise an exception|
|Report suppression of an error without raising an exception (e.g. error handler in a long-running server process)|logging.error(), logging.exception() or logging.critical() as appropriate for the specific error and application domain|

### Level of logging

|Level|When it's used|
| :----- | :----- |
|DEBUG|Detailed information, typically of interest only when diagnosing problems.|
|INFO|Confirmation that things are working as expected.|
|WARNING|An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.|
|ERROR|Due to a more serious problem, the software has not been able to perform **some function.**|
|CRITICAL|A serious error, indicating that the program itself may be unable to continue **running.**|

### Config
basicConfig()

```python
logging.basicConfig(filename='myapp.log', format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
```
only the first call will actually do anything: subsequent calls are effectively no-ops.

### Basic calling
```python
import logging
logging.debug()
logging.warning()
logging.info()
logging.error()
logging.critical()
```
## Advanced Logging Tutorial
logging库包含4 components:

* Logge类暴露打log的接口。
* Handler类以不同的方式处理log，输出到不同的终端（例如stdout，文件，email地址等）。
* Filter类能更细粒度的过滤log，而不仅依靠level过滤。
* Formatter格式化log。

通过调用Logger对象实例的方法，来开展logging过程，整个过程在处理和传递`LogRecord` 对象。
Logger对象实例有name，每个Logger对象实例之间有层级关系，通过name的'.'来表示层级关系。
惯例上推荐使用module-level logger，命名规范如下：

```python
logger = logging.getLogger(__name__)
```
日志配置三种方式：

* 显式的构造logger，handlers，formatters
* 创建配置文件，使用fileConfig()方法
* 创建配置dict，传递给dictConfig()方法

### Logging Flow
其中filter既可以和logger绑定，用来过滤日志到handler，也可以和handler绑定，用来过滤日志到formatter。

![image](images/41ff0659-8962-434c-9013-226c71c784ac.png)

### Logger
作用：

1. 提供应用程序调用接口
2. 根据 level 或者 filter 确定哪些message需要操作
3. 传递 message 给 handler

通过logging.getLogger实例化

logger 通过name的dot来建立层级关系，所以习惯上推荐以module.\_\_name\_\_命名logger。
For example, a logger named ‘scan’ is the parent of loggers ‘scan.text’, ‘scan.html’ and ‘scan.pdf'.
层级关系的好处：
某些属性，比如level，handles等，当子logger没有设置时，会向上传递，直到搜索到对应的设置，最多到root logger。
这样，只需要配置root logger，就可以达到全局配置。
当然，子logger可以单独配置。

一键配置：`logging.basicConfig(...)`

手动配置：
Logger.setLevel() specifies the lowest-severity log message a logger will handle.
Logger.addHandler() and Logger.removeHandler() add and remove handler objects from the logger object.
Logger.addFilter() and Logger.removeFilter() add and remove filter objects from the logger object.

调用：
Logger.debug(), Logger.info(), Logger.warning(), Logger.error(), and Logger.critical() 
Logger.exception() creates a log message similar to Logger.error(). The difference is that Logger.exception() dumps a stack trace along with it. Call this method only from an exception handler.
Logger.log() takes a log level as an explicit argument. 

### LogRecord
由logger自动生成，日志的载体。日志可以包括时间、级别、文件、消息（message）等等信息。

`makeLogRecord(attrdict)` 可手动生成。

相当于一个`namedtuple` ，各属性可以被access。

重要属性：

* name(str)
* level(int)
* pathname(str)
* msg(Any): 通常是带占位符的%-string，够成消息
* args(tupe|dict)：用于填充%-string的数据，够成消息

获取消息：`record.getMessage()` 

### Handler
Handler负责将log信息输出到指定终端上，例如标准输出、文件或电子邮件等。

Logger对象可以有多个Handler，例如，某应用需要将全部log信息保存，将error级别的信息打印到stdout，将critical级别信息发送到指定e-mail地址。该场景需要add三个Handler，并根据Level将log信息分别处理。

handler都继承自共同的基类`Handler`，我们使用或定义继承handler的子类。

handler的常用子类：

1. `StreamHandler`：将信息发送到流，默认是标准错误*sys.stderr* ;
2. `FileHandler`：将信息发送到文件；
3. `SocketHandler` ：将信息发送到TCP/IP sockets；
4. `SMTPHandler` ：将信息发送到指定的Email；
5. `HTTPHandler` ：将信息发送到HTTP server；

开发者相关配置：
setLevel()：设置handler要处理信息的级别，高于这个级别的信息才会往后传递进一步处理；
setFormatter()：设置这个handler要使用的formatter
addFilter() and removeFilter()：设置或取消这个handler要用的filter

和`Logger`关联：`addHandler()` 

### Formatter
设置消息的顺序、结构和内容。

```python
logging.Formatter.__init__(fmt=None, datefmt=None, style='%')
```
默认格式是%-string，如："%(asctime)s:%(levelname)s:%(message)s"

可以是f-string, 如："{asctime}:{levelname}:{message}"

传参为空，均有默认处理方式。

补习下 %-字符串：

C-风格的字符串格式化方式，类似f-string。

可以按位置格式化多个变量：

```python
# This prints out "John is 23 years old."
name = "John"
age = 23
print("%s is %d years old." % (name, age))
```
也可以按名称格式化多个变量：

```python
# This prints out "John is 23 years old."
name = "John"
age = 23
print("%(name)s is %(age)d years old." % {'name': name, 'age': age})
```
fmt格式化字符串里待填充的信息，会通过`LogRecord` 的属性获取。

和handler关联：`handler.setFormatter(formatter)`

### Filter
filter用于提供比级别更复杂的消息过滤依据。

所谓的filter不用真的是一个`Filter` 对象，实际上，它只要是一个具有`filter(record)` 方法的对象或一个参数是record的方法就可以。所以关键接口：

```python
class A:
    def filter(self, record):
        ...  # return 0 for no, nonzero for yes

# any callable
def foo(recor):
    ...  # return 0 for no, nonzero for yes
```
既可以和logger关联: `logger.adddFilter(filter)`

也可以和handler关联: `handler.addFilter(filter)` 

### Example
```python

import logging

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
```