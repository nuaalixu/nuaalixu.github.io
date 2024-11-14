---
layout: post
category: Python编程
---

# import、module和package
## Package vs Module
模块是最小的代码组织单元，包是一组模块的集合。

Module是Code的集合（亦是namespace），Package是Module的集合（亦是namespace）。

Module和Package都可以被import。但Module被import时，代码可见，Package被import时，只有`__init__.py`的代码可见。

Module可以被直接执行，Package被间接执行，实际执行的是`__main__.py`的代码。

```
package
├── __init__.py
├── main.py
├── subpackage1
│   └── module1.py
└── subpackage2
    └── module2.py
```
目录层级结构是客观存在的，层级关系用`.` 号表示。

在运行时，package和module必须先import后才能被引用。

但Module被import时，代码可见，Package被import时，只有`__init__.py`的代码可见。

如下所示，虽然`pacakge` 被引入了，但是其下的`subpackage1` 无法被直接索引到。

```python
>>> import package
>>> package.subpackage1
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'package' has no attribute 'subpackage1'
```
但是其中的类和函数可以通过`.` 索引到。

```python
>>> # 类A在__init__.py中被定义
>>> package.A
<class 'package.A'>
```
这是因为，首次引入`package`后，只执行了`__init__.py`的代码内容，所以`pacakge`的namespace中只有类A，是没有`subpackage1` 和`subpackage2`的。

```python
>>> dir(package)
['A', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']
```
通过`import`子包/模块，`package`的namespace会相应增加。

```python
>>> import package.subpackage1.module1
>>> dir(package)
['A', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'subpackage1']
```
## Modules vs Scripts
同一个code文件，在不同侧重的用法时，有时被称作module，有时称作script。

**通常modules**被用于**imported，scripts被用于直接执行。**

希望直接执行module时，需要使用解释器的`-m` 选项。

`-m` 选项在`sys.path` 中搜索module名称，然后以`__main__` 作为module的名称执行代码。

## sys.path
执行python脚本时，脚本所在路径会加到`sys.path`中。

```bash
python $dir/script.py ...
```
所以在非包调用时，脚本`script.py` 如果需要其他`$dir`下的其他module，可以直接import。

```python
# script.py
import submodule
...
```
# 