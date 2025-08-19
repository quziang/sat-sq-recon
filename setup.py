
# 版权所有 (c) 2023 SLAB Group
# 作者: Tae Ha "Jeff" Park (tpark94@stanford.edu)
#
# 本文件用于项目的打包和 Cython 扩展模块的编译。


# 导入 distutils 的 setup 方法，用于打包 Python 项目
from distutils.core import setup
# 导入 cythonize 用于编译 Cython 文件
from Cython.Build import cythonize
# 导入 Extension 用于定义扩展模块
from distutils.extension import Extension


# 导入 numpy，用于获取头文件路径
import numpy as np


# 定义 setup_package 函数，执行项目的打包和扩展模块编译
def setup_package():
    setup(
        name="sampler",  # 包名
        ext_modules=cythonize([
            Extension(
                "core.utils.libmesh.triangle_hash",  # 扩展模块名称
                [
                    "core/utils/libmesh/triangle_hash.pyx"  # Cython 源文件路径
                ],
                language="c++11",  # 使用 C++11 标准
                libraries=["stdc++"],  # 链接标准 C++ 库
                include_dirs=[np.get_include()],  # 包含 numpy 头文件
                extra_compile_args=["-std=c++11", "-O3"]  # 编译参数：C++11 和优化
            ),
        ])
    )


# 主入口，执行打包函数
if __name__ == "__main__":
    setup_package()
