---
title: cJSON源码学习笔记
date: 2025-12-08 17:38:16
categories:
  - 技术
tags:
  - C语言
---

### Day 1

```c
#ifndef cJSON__h
#define cJSON__h
```

避免头文件重复定义。首先检查是否已经包含了 `cJSON__h` 宏，如果没有包含则包含之并执行下面的代码；如果已经包含则跳过这段代码，避免了同一份文件的重复执行，导致同一个文件内类和结构体等被多次定义等问题（至少造成了编译时间增加）。对应的 `#endif` 在文件的末尾。

```c
#ifdef __cplusplus
extern "C"
{
#endif
```

`__cplusplus` 是 g++ 编译器定义的宏，这段话就是说，如果使用 C++ 编译器编译这个文件，就执行 `extern "C" {}` 代码，把全文包含进去。

**extern 关键字：** 全局变量默认具有外部链接属性（在外部需要使用 extern 关键字声明），如果用 static 修饰就变成了内部链接属性（不能在外部调用）。函数默认具有外部链接属性，可以在外部直接调用，static 修饰就变成了内部链接属性。

> 为什么要在头文件中 extern？
> 头文件是声明变量而不是定义变量的地方，我们通过 extern 把变量的作用域从定义变量的文件扩展到了头文件，进而扩展到了包含头文件的源文件，使得变量在源文件可用。

**extern "C"**： 首先起到了 extern 的作用。其次，当 C++ 编译器编译代码时，会进行 **函数名修饰**。因为 C++ 支持函数重载，编译器会把 `void print(int x)` 和 `void print(double x)` 在内部改写成不同的名字，比如 `_Z3printi` 和 `_Z3printd`。但是 C 不支持函数重载，所以 `void print(int x)` 的名字就是 `_print`。这导致用 C 编译器编译的文件和 C++ 编译器编译的文件链接时，可能出现“找不到函数”的问题。用 extern "C" 修饰的函数会强制按照 C 语言的方式来编译，以避免链接的时候找不到对应的函数，出现错误。

```c
#if !defined(__WINDOWS__) && (defined(WIN32) || defined(WIN64) || defined(_MSC_VER) || defined(_WIN32))
#define __WINDOWS__
#endif
```

把不同的 Windows 版本的宏定义统一成 `__WINDOWS__`

### Day 2

```c
#ifdef __WINDOWS__

#define CJSON_CDECL __cdecl
#define CJSON_STDCALL __stdcall
```

定义了两种 C 语言中的调用约定，`__cdecl` 是默认的调用约定，规定由调用者清理栈帧；`__stdcall` 是 Windows API 的调用约定，规定由被调用者清理栈帧。

> 调用者清理：主要是 cdecl，即 C 语言的默认调用约定。规定参数从右往左入栈，调用者清理栈帧，使得可变长参数得以实现，函数名前缀\_（函数名修饰）；
> 被调用者清理：Pascal 规约是 Pascal 语言的默认调用约定，参数从左往右入栈，被调用者清理栈帧；stdcall 是微软的调用约定，参数从右往左入栈，被调用者清理栈帧，编译的函数名前缀\_，后缀@及栈空间长度。这种约定不允许可变参数，但是更有效率，解堆栈的代码不需要每次调用时生成一遍。

```c
/* export symbols by default, this is necessary for copy pasting the C and header file */
#if !defined(CJSON_HIDE_SYMBOLS) && !defined(CJSON_IMPORT_SYMBOLS) && !defined(CJSON_EXPORT_SYMBOLS)
#define CJSON_EXPORT_SYMBOLS
#endif
```

如果没有定义 `CJSON_HIDE_SYMBOLS` 和 `CJSON_IMPORT_SYMBOLS`，就定义 `CJSON_EXPORT_SYMBOLS`（默认）。

- `CJSON_HIDE_SYMBOLS`：不导出任何符号，一般用于构建静态库；
- `CJSON_EXPORT_SYMBOLS`：把符号导出到 dll 中，供用户使用；
- `CJSON_IMPORT_SYMBOLS`：客户端程序声明从 dll 中导入符号。

```c
#if defined(CJSON_HIDE_SYMBOLS)
#define CJSON_PUBLIC(type)   type CJSON_STDCALL
#elif defined(CJSON_EXPORT_SYMBOLS)
#define CJSON_PUBLIC(type)   __declspec(dllexport) type CJSON_STDCALL
#elif defined(CJSON_IMPORT_SYMBOLS)
#define CJSON_PUBLIC(type)   __declspec(dllimport) type CJSON_STDCALL
#endif
#else /* !__WINDOWS__ */
#define CJSON_CDECL
#define CJSON_STDCALL
```

调用 Windows API `__declspec(dllexport)` 等进行具体实现。最后定义空宏，保证在其它平台上宏仍然存在但无实际意义。

注意到 EXPORT 和 IMPORT 分别是属于开发者和客户端的行为，但都定义在同一个头文件中，这体现了头文件的共享性。开发者在构建静态库或者动态链接库的时候，可以通过编译指令定义 `CJOSN_EXPORT_SYMBOLS`，这样代码中所有签名为 `CJSON_PUBLIC(int) func(...)` 的函数就被展开为 `__declspec(dllexport) int func(...)`，从而编译器知道要将它们导出到 dll 中。而客户端在编译的时候可以指定 `CJSON_IMPORT_SYMBOLS`，并包含头文件，则头文件中的 `CJSON_PUBLIC(int)` 就会替换成 `__declspec(dllimport) int`，编译器自动去 dll 中寻找对应的代码。

```c
#if (defined(__GNUC__) || defined(__SUNPRO_CC) || defined (__SUNPRO_C)) && defined(CJSON_API_VISIBILITY)
#define CJSON_PUBLIC(type)   __attribute__((visibility("default"))) type
#else
#define CJSON_PUBLIC(type) type
#endif
#endif
```

在支持 `__attribute__((visibility("default")))` 的编译器中，通过显式声明控制符号对外可见；如果当前环境不支持，则不做任何修饰。

```c
#define CJSON_VERSION_MAJOR 1
#define CJSON_VERSION_MINOR 7
#define CJSON_VERSION_PATCH 18
```

定义版本号，用户可以通过 `#if (CJSON_VERSION_MAJOR >= 1 && CJSON_VERSION_MINOR >= 7) #else #endif` 这样的条件编译来启用新版本特性和兼容旧版本。

```c
/* cJSON Types: */
#define cJSON_Invalid (0)
#define cJSON_False  (1 << 0)
#define cJSON_True   (1 << 1)
#define cJSON_NULL   (1 << 2)
#define cJSON_Number (1 << 3)
#define cJSON_String (1 << 4)
#define cJSON_Array  (1 << 5)
#define cJSON_Object (1 << 6)
#define cJSON_Raw    (1 << 7) /* raw json */

#define cJSON_IsReference 256
#define cJSON_StringIsConst 512
```

定义基本类型，使用位运算的好处是可以高效地进行判断，比如 `if (item->type & cJSON_String)`。

```cpp
typedef struct cJSON
{
    /* next/prev allow you to walk array/object chains. Alternatively, use GetArraySize/GetArrayItem/GetObjectItem */
    struct cJSON *next;
    struct cJSON *prev;
    /* An array or object item will have a child pointer pointing to a chain of the items in the array/object. */
    struct cJSON *child;

    /* The type of the item, as above. */
    int type;

    /* The item's string, if type==cJSON_String  and type == cJSON_Raw */
    char *valuestring;
    /* writing to valueint is DEPRECATED, use cJSON_SetNumberValue instead */
    int valueint;
    /* The item's number, if type==cJSON_Number */
    double valuedouble;

    /* The item's name string, if this item is the child of, or is in the list of subitems of an object. */
    char *string;
} cJSON;
```

- next 和 prev 把同一层级的节点构建成双向链表；
- 如果这个位置的节点是一个对象或者数组，child 指向对象或数组的第一个节点（元素）；
- type 是上面的 6 种基本类型之一；
- valuestring, valueint, valuedouble 存储当前节点的数据；当前节点数据是什么类型的，就存储在哪个变量中，剩下两个为空；如果都不是，则都为空；
- string 存储键名。

举例说明，如果有以下的 JSON，则：

```json
{
  "key1": "value1",
  "key2": {
    "nestedKey": "nestedValue"
  },
  "key3": [1, 2, 3]
}
```

- 根节点的 child 指向 key1；
- key1, key2, key3 组成双向链表；
- key1 节点的 type 为 CJSON_String， string 为 "key1"，valuestring 为 "value1"；
- key2 节点的 child 指向 nestedKey，key3 节点的 child 指向数组第一个元素 1;
- 因为 cJSON 的思想是把所有数据类型都统一成 cJSON 节点，所以数组中的元素也是一个个 cJSON 节点。

cJSON 的链表结构便于增删节点，无需重新分配内存。

### Day 3

```c
typedef struct cJSON_Hooks
{
      /* malloc/free are CDECL on Windows regardless of the default calling convention of the compiler, so ensure the hooks allow passing those functions directly. */
      void *(CJSON_CDECL *malloc_fn)(size_t sz);
      void (CJSON_CDECL *free_fn)(void *ptr);
} cJSON_Hooks;

typedef int cJSON_bool;
```

cJSON_Hooks 结构体提供了一个接口，允许实现自定义的内存管理函数。`*malloc_fn` 传入分配的内存大小，返回分配的内存地址；`*free_fn` 传入要释放的内存指针。两个都强制声明为 `CJSON_CDECL` 的，保证在 Windows 平台下 malloc 和 free 由调用者清理栈帧（与标准库的实现统一）。

cJSON_bool 是自定义的布尔类型。在早期的 C 语言中没有布尔类型，为了统一行为，使用 int 自定义布尔类型。

```c
/* Limits how deeply nested arrays/objects can be before cJSON rejects to parse them.
 * This is to prevent stack overflows. */
#ifndef CJSON_NESTING_LIMIT
#define CJSON_NESTING_LIMIT 1000
#endif
```

cJSON 通过递归的方式来处理嵌套，而每次递归都需要压栈。如果嵌套层数过多，可能引发栈溢出，所以有必要定义一个嵌套上限。

```c
/* Limits the length of circular references can be before cJSON rejects to parse them.
 * This is to prevent stack overflows. */
#ifndef CJSON_CIRCULAR_LIMIT
#define CJSON_CIRCULAR_LIMIT 10000
#endif
```

限制循环引用同样是为了避免爆栈。原生 JSON 中不支持引用，如果引入了 ref 则可能出现循环引用，如：`{"a": {"b": {"c": {"ref": "$a"}}}}` 而其他依赖于引用计数的语言可能因为循环引用而无法垃圾回收。
