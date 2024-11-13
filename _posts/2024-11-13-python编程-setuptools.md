---
layout: post
category: Python编程
---

# setuptools
## Entry Points
入口点是包的元数据（metadata），在包安装时暴露。作用于两个场景：

1. 提供可在终端执行的命令，即*console script*，比如pip包提供了可直接运行的`pip` 命令。
2. 提供了通过插件实现定制功能的能力，比如pytest包允许通过`pytest11` 入口点接入插件。

Python定义了入口点对象即`EntryPoint`，每个 `EntryPoint` 都有一个 `.name`、`.group` 和 `.value` 属性以及一个 `.load()` 方法来解析值。

### Console Scripts
假设包的目录结构如下：

```bash
project_root_directory
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins
        ├── __init__.py
        └── ...
```
其中`__init__.py` 包含一个函数：

```python
def hello_world():
    print("Hello world")
```
如何想要命令行执行`hello_word()` ，方法之一是在创建`src/timmins/__main__.py` 文件，内容代码调用该方法：

```python
from . import hello_world

if __name__ == '__main__':
    hello_world()
```
然后就可以通过`python -m` 调用：

```bash
$ python -m timmins
Hello world
```
而通过入口点，可以便捷地创建一个可执行命令，即console script。入口点配置如下：

```python
# setup.py文件
from setuptools import setup

setup(
    # ...,
    entry_points={
        'console_scripts': [
            'hello-world = timmins:hello_world',
        ]
    }
)
```
上例以`setup.py`文件为例，还可以通过`pyproject.toml`和`setup.cfg`配置。

这里的`console_scripts`是入口点的组名（group），表明该命令属于`console_scripts` 组，组名可以被其他包如`pip` 辨别。`console_scripts` 即告诉`pip`包将该组的所有方法包装成可执行命令。

`hello-world` 是入口点的名字（name），即终端调用的命令名；

`timmins:hello_world`是入口点的值（value），表明该命令实际执行timmins包的`hello_world`方法。

在包安装后，就可以直接调用名为`hello-world`的命令：

```bash
$ hello-world
Hello world
```
### Entry Points for Plugins
入口点允许一个包公开其某些功能，以供其他库和应用程序发现并使用。这一特性使得包可以通过插件扩展和定制功能。

例如，上述timmins包的目录结构不变：

```bash
project_root_directory
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins
        ├── __init__.py
        └── ...

```
`src/timmins/__init__.py` 内容修改如下：

```python
def display(text):
    print(text)

def hello_world():
    display('Hello world')
```
此时`hello_world()`方法调用`display()` 方法打印文本。

如果我们希望使用不同的打印风格，只需要修改`display()`方法即可。那么如何通过插件的方式，提供不同的`display()`方法供`hello_world()`调用呢？

包和插件之间通过入口点这一协议进行交互。

我们实现一个插件包给timmins提供不同的`display()`方法，插件包目录结构如下：

```bash
timmins-plugin-fancy
├── pyproject.toml        # and/or setup.cfg, setup.py
└── src
    └── timmins_plugin_fancy
        └── __init__.py
```
在`src/timmins_plugin_fancy/__init__.py` ，定义新的`display()`方法：

```python
def excl_display(text):
    print('!!!', text, '!!!')
```
为了让timmins包能“发现”该插件提供的功能，需要修改插件包配置文件`setup.py` ，也可以通过`pyproject.toml`和`setup.cfg`配置。

```python
# setup.py
from setuptools import setup

setup(
    # ...,
    entry_points = {
        'timmins.display': [
            'excl = timmins_plugin_fancy:excl_display'
        ]
    }
)
```
这里`timmins.display` 作为组名标志，可以被其他包“发现“。

对应timmins包也需要修改，将原本调用脚本`src/timmins/__init__.py`的内容改为：

```python
from importlib.metadata import entry_points
display_eps = entry_points(group='timmins.display')
try:
    display = display_eps[0].load()
except IndexError:
    def display(text):
        print(text)

def hello_world():
    display('Hello world')
```
其中`importlib.metadata.entry_points` 方法会检查所有安装包的元数据，即`dist-info` 或 `egg-info` 目录，收集其中的入口点。

此处`entry_points(group='timmins.display')` 指定收集组名为`timmins.display`的入口点。

因为timmins包的`display` 方法被插件timmins-plugin-fancy包提供的方法替代，所以此时执行：

```bash
$ hello-world
!!! Hello world !!!
```
### Entry Points Syntax
入口点的配置语法：

```python
<name> = <package_or_module>[:<object>[.<attr>[.<nested-attr>]*]]
```
入口点值的解析相当于通过import代码进行解析。例如：

```yaml
<name> = <package_or_module>
```
相当于：

```python
import <package_or_module>
parsed_value = <package_or_module>
```
例如：

```yaml
<name> = <package_or_module>:<object>
```
相当于

```python
from <package_or_module> import <object>
parsed_value = <object>
```
例如：

```python
<name> = <package_or_module>:<object>.<attr>.<nested_attr>
```
相当于：

```python
from <package_or_module> import <object>
parsed_value = <object>.<attr>.<nested_attr>
```