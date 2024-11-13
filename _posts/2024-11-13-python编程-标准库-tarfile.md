---
layout: post
category: Python编程
---

# 标准库: tarfile
## tar文件
“tar”源自于“tape archive”，它最初是为了将数据写入不带文件系统的顺序 I/O 设备（例如使用磁带的设备）而开发的。

基本原理：

GNU tar的基本原理是将多个文件和目录组合成一个单一的文件。它将这些文件和目录打包成一个归档（archive），并将其存储为一个连续的数据流。归档文件保留了文件和目录的层次结构和元数据（如权限、所有者、时间戳等），可以通过解压缩归档文件来还原原始的文件和目录结构。

数据格式：

tar 对于归档文件格式来说是很不寻常的。她没有归档头，没有文件索引来方便查找，没有魔术字节来帮助file检测文件是否为 tar，没有页脚，没有归档范围的元数据。

GNU tar的数据格式是一种顺序存储的二进制格式，它由一系列文件对象组成，而文件对象又由文件头和数据载荷构成。

典型的文件对象的文件头：描述一个普通文件的元数据，包括文件名、权限、所有者、组、大小、时间戳等信息。

```cpp
struct file_header {
	char file_path[100];
	char file_mode[8];
	char owner_user_id[8];
	char owner_group_id[8];
	char file_size[12];
	char file_mtime[12];
	char header_checksum[8];
	char file_type;
	char link_path[100];

	char padding[255];
};
```
接下来是 `ceil(file_size / 512)` 512 字节的数据载荷块（即文件内容）。

此外，tar 文件应该以 1024 个零字节结尾，以表示文件结束标记。

例如，我们要打包一个文件“hello.txt“，文件内容是“hello world”，需要两个512字节块。

1. Bytes 0-511: Header, `type='0'`, `file_path="./hello.txt"`, `file_size=11`
2. Bytes 512-1023: `"Hello World"`, followed by 501 zero bytes

## tarfile库
支持格式：tar，包括gzip, bz2 和 lzma

基本架构：

* `tarfile.open`一般是入口，打开tar achieve文件，返回一个`TarFile`对象。
* `TarFile`对象提供了操作tar achieve的API，是多个文件对象的**序列，支持****with****语义。**每个文件对象由`TarInfo`表示。
* `TarInfo` 代表包中的一个文件对象，提供文件对象元信息访问和文件类型确认方法。但不包括数据载荷本身。

#### tarifle.open(*name=None*, *mode='r'*, *fileobj=None*, *bufsize=10240*, *\*\*kwargs*)
返回一个指向name路径的`TarFile` 对象。

重点关注下文件打开模式*mode*，有两种形式：

   * `'filemode[:compression]'` ：常规方式，支持追加写入，如`'r'`或`'r:*'` ，`'r:gz'` ，`'x'` 或 `'x:'`，`'a'` 或 `'a:'` ，`'w:gz'` 等
   * `'filemode|[compression]'` ：表示以文件流的方式打开，此时不支持随机寻址，但*fileobj*可以是任意文件流对象，覆盖*name*。如：`'r|*'` ，`'r|'` ，`'r|gz'` ，`'w|gz'`等。

#### *class* tarfile.TarFile(*name=None*, *mode='r'*, *fileobj=None*, *format=DEFAULT\_FORMAT*, *tarinfo=TarInfo*, *dereference=False*, *ignore\_zeros=False*, *encoding=ENCODING*, *errors='surrogateescape'*, *pax\_headers=None*, *debug=0*, *errorlevel=1*)
代表整个tar文件，提供其中文件对象的访问接口。

常用接口：

   * TarFile.getmember(*name*)：返回包中指定name的成员，即`TarInfo`对象。
   * TarFile.getmembers()：返回包的全部成员的列表，即`TarInfo` 的列表。
   * TarFile.next()：返回包中下一个`TarInfo` 成员。
   * TarFile.extractall(*path='.'*, *members=None*, *\**, *numeric\_owner=False*, *filter=None*)：提取包中所有的文件到指定路径，默认当前路径。*filter*字符串指定提取文件时的过滤方式，以防止恶意攻击。
   * TarFile.extract(*member*, *path=''*, *set\_attrs=True*, *\**, *numeric\_owner=False*, *filter=None*)：提取包中指定成员到指定路径，*member*可以是路径名或`Tarinfo`对象。
   * TarFile.extractfile(*member*)：提取指定成员返回一个`io.BufferedReader`对象。
   * TarFile.add(*name*, *arcname=None*, *recursive=True*, *\**, *filter=None*)：添加指定文件到tar包中。
   * TarFile.addfile(*tarinfo*, *fileobj=None*)：添加`TarInfo` 对象到包中，如果提供了*fileobj*，则从中读取`TarInfo.size`字节的数据到包中。
   * TarFile.close()：关闭包文件，写模式下，在包文件末尾添加1024字节的0数据块。

#### *class* tarfile.TarInfo(*name=''*)
代表包中的一个文件对象，提供描述信息和快捷的类型判断方法。

常用属性：

   * TarInfo.name: str
   * TarInfo.size: int
   * TarIfo.mtime: int|float，最近一次修改时间。
常用方法：
   * TarInfo.isfile()/TarInfo.isreg()
   * TarInfo.isdir()

## Examples
简单的提取全部

```python
import tarfile
tar = tarfile.open("sample.tar.gz")
tar.extractall(filter='data')
tar.close()
```
简单的打包

```python
import tarfile
with tarfile.open("sample.tar", "w") as tar:
    for name in ["foo", "bar", "quux"]:
        tar.add(name)
```
查看每个成员的信息

```python
import tarfile
tar = tarfile.open("sample.tar.gz", "r:gz")
for tarinfo in tar:
    print(tarinfo.name, "is", tarinfo.size, "bytes in size and is ", end="")
    if tarinfo.isreg():
        print("a regular file.")
    elif tarinfo.isdir():
        print("a directory.")
    else:
        print("something else.")
tar.close()
```