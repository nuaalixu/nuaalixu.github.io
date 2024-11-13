---
layout: post
category: Python编程
---

# 标准库-pathlib
## 基本架构
路径类分成两类：

* 纯路径/pure path：仅用于逻辑操作，比如：路径的拼接，basename等，本质上都是字符串操作；
* 具体路径/concrete path：不仅用于逻辑操作，还涉及实际IO操作（和OS
交互），比如创建目录，检查是否存在等；

类图如下：

![image](/images/pSGZ3-ctIZK0TytFiGL3r1V5cjvYrraZ2c6Jy43aMOY.png)

Path类是具体路径，可以根据不同平台使用更具体的子类，也可以使用Path作为跨平台的统一接口。一般情况下，Path是最通用的推荐接口。

PurePath是纯路径，不同平台的路径格式有区别，所以PurePath也有针对具体平台的子类。

需要注意，平台相关的具体路径在不匹配的平台上无法实例化，但是平台相关的纯路径，比如PurePosixPath在Windows上也可以实例化，因为只涉及逻辑操作，不和OS交互。

## 基本使用
以最通用的Path类为例。

列出目录内容

```python
p = Path('.')
p.iterdir()
```
通配符模式查找文件：使用`glob`

```python
p.glob('**/*.py') # 递归查找
p.rglob('*.py') # 递归查找
```
路径拼接：使用`/`

```python
p / 'init.d' / 'apache2'
p.joinpath('init.d', 'apache2') # 等价
```
查询路径属性

```python
q.exists()
q.is_dir()
q.is_file()
```
路径名逻辑操作：使用路径属性获取

```python
p.root # 根目录
p.parent # 父目录
p.name # basename
p.suffix # 最后一个后缀
p.suffixes # 全部后缀
p.stem # 去掉后缀的basename
```
创建目录

```python
p.mkdir()
```
获取绝对路径

```python
p.absolute()
```
打开文件

```python
with q.open() as f: f.readline()
```
## 高级使用
路径名替换

```python
p = PureWindowsPath('c:/Downloads/pathlib.tar.gz')
p.with_name('setup.py') # PureWindowsPath('c:/Downloads/setup.py')
p.with_stem('lib') # PureWindowsPath('c:/Downloads/lib.gz')
p.with_suffix('.bz2') # PureWindowsPath('c:/Downloads/pathlib.tar.bz2')
```