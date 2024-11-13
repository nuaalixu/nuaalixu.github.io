---
layout: post
category: Python编程
---

# 标准库: contextlib
## with语句
`with`语句用于包装带有使用上下文管理器定义的方法的代码块的执行。

带有一个“项目”的 `with` 语句的执行过程如下:

1. 对上下文表达式进行求值来获得上下文管理器。
2. 载入上下文管理器的 `__enter__()` 以便后续使用。
3. 载入上下文管理器的 `__exit__()` 以便后续使用。
4. 发起调用上下文管理器的 `__enter__()` 方法。
5. 如果 `with`语句中包含一个目标，来自 `__enter__()` 的返回值将被赋值给它。
6. 执行语句体。
7. 发起调用上下文管理器的 `__exit__()` 方法。 如果语句体的退出是由异常导致的，则其类型、值和回溯信息将被作为参数传递给 `__exit__()`。 否则的话，将提供三个 `None`参数。

如果语句体的退出是由异常导致的，并且来自 `__exit__()` 方法的返回值为假，则该异常会被重新引发。 如果返回值为真，则该异常会被抑制，并会继续执行 `with` 语句之后的语句。

如果语句体由于异常以外的任何原因退出，则来自 `__exit__()` 的返回值会被忽略，并会在该类退出正常的发生位置继续执行。

`with` 语句在语义上等价于：

```python
manager = (EXPRESSION)
enter = type(manager).__enter__
exit = type(manager).__exit__
value = enter(manager)
hit_except = False

try:
    TARGET = value
    SUITE
except:
    hit_except = True
    if not exit(manager, *sys.exc_info()):
        raise
finally:
    if not hit_except:
        exit(manager, None, None, None)
```
## 上下文管理器类
上下文管理器类实现了两个特殊的方法`__enter__`和`__exit__`，允许用户自定义运行时上下文，在`with` 语句体被执行前调用`__enter__` 方法，在语句体结束后调用`__exit__` 方法。

`contextmanager.__enter__()` : 

进入运行时上下文，返回自身或跟运行时上下文相关的对象，返回值可与`as` 语句指定的变量绑定。

例如，文件对象通过`__enter__` 方法返回自身，从而允许`open()`方法被`with` 语句使用。

而`decimal.localcontext()`的`__enter__` 方法返回的是小数管理器的副本（不是自身），从而可以在上下文中调整当前小数精度等配置，而不影响`with` 语句体外的配置，并在退出上下文后恢复。

`contextmanager.__exit__(`*exc\_type*`, `*exc\_val*`, `*exc\_tb*`)`：

退出运行时上下文，并且返回布尔值用于通知`with`语句是否抑制异常。

当`with`语句体发生异常时，调用上下文管理器的`__exit__` 方法，并将异常类型、异常值和回溯信息作为参数传递，`__exit__` 方法可以进行异常处理（上下文善后操作），但无法阻止异常传递给`with`语句，只能通过返回值是True还是False，来通知`with`语句继续抛出异常，还是抑制异常。当返回值是False时，既可以向上传递异常，又可以表明`__exit__` 方法执行完毕。

## contextlib
contextlib标准库提供了`with`语句相关的工具。下面列举一些实用的工具。

`@contextlib.contextmanager`

该装饰器可以定义一个支持`with`语句上下文管理器的工厂函数，而不需要创建一个类支持单独的`__enter__`和`__exit__` 方法。

被装饰的函数在被`()`调用时，必须返回一个生成器，该生成器只能`yield`一个对象。

当执行`with` 和被装饰函数组成的语句时，首先调用被装饰函数，得到一个生成器；然后执行生成器，得到一个被`yield` 出来的对象，此对象可用于`as`语句绑定别名；接着执行上下文语句体内的代码；最后恢复到生成器里，执行生成器`yield`后的代码。如果上下文语句体中发生了未被处理的异常，则该异常会在恢复到生成器里重新被引发，因此，你可以使用 try...except...finally语句来捕获该异常（如果有的话），或确保进行了一些清理。

下面是一个抽象示例：

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(*args, **kwds):
    # Code to acquire resource, e.g.:
    resource = acquire_resource(*args, **kwds)
    try:
        yield resource
    finally:
        # Code to release resource, e.g.:
        release_resource(resource)


with managed_resource(timeout=3600) as resource:
    # Resource is released at the end of this block,
    # even if code in the block raises an exception
```
`contextlib.closing(thing)`

返回一个在语句块执行完成时关闭 *things* 的上下文管理器。这基本上等价于：

```python
from contextlib import contextmanager

@contextmanager
def closing(thing):
    try:
        yield thing
    finally:
        thing.close()
```
实际使用时，允许你编写这样的代码：

```python
from contextlib import closing
from urllib.request import urlopen

with closing(urlopen('https://www.python.org')) as page:
    for line in page:
        print(line)
```
在执行完`with`语句块之后，会调用`page.close()` ,即使发生异常也能顺利关闭。

`contextlib.suppress(`*\*exceptions*`)`

返回一个当指定的异常在 `with` 语句体中发生时会屏蔽它们然后从 `with` 语句结束后的第一条语言开始恢复执行的上下文管理器。

与完全抑制异常的任何其他机制一样，该上下文管理器应当只用来抑制非常具体的错误，并确保该场景下静默地继续执行程序是通用的正确做法

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove('somefile.tmp')

with suppress(FileNotFoundError):
    os.remove('someotherfile.tmp')
```
等价于：

```python
try:
    os.remove('somefile.tmp')
except FileNotFoundError:
    pass

try:
    os.remove('someotherfile.tmp')
except FileNotFoundError:
    pass
```
`contextlib.redirect_stdout(`*new\_target*`)`

用于将 `sys.stdout` 临时重定向到一个文件或类文件对象的上下文管理器。

该工具给已有的将输出硬编码写到 stdout 的函数或类提供了额外的灵活性。

```python
with redirect_stdout(io.StringIO()) as f:
    help(pow)
s = f.getvalue()
```
 `help()`的输出通常会被发送到 `sys.stdout`。 你可以通过将输出重定向到一个 `io.StringIO`对象来将该输出捕获到字符串。 替换的流是由 `__enter__` 返回的因此可以被用作 `with`语句的目标。