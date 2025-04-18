# mlir-test

Pyenv installation for fedora:

https://joepreludian.medium.com/starting-your-python-dev-environment-with-pyenv-and-pipenv-on-a-redhat-gnu-linux-based-system-d66795377ea 
sudo yum install libffi-devel zlib-devel bzip2-devel readline-devel sqlite-devel wget curl llvm ncurses-devel openssl-devel lzma-sdk-devel libyaml-devel lzma xz-devel (in webpage it installed redhat-rpm- package, I don’t think it is necessary for me)
Use venv to build in vscode and cmake
1.	Python -m venv mlir_venv
2.	Command palette: Python select interpreter
3.	Command palette: Cmake Edit local user kits
4.	Add this "environmentSetupScript": "/home/qzhuang/code/mlir_venv/bin/activate"
5.	Command palette: Cmake select a kit
6.	Command palette: Cmake delete cache and rebuild

Current LLVM and Torch-mlir base:
LLVM: 72144d119a7291f8b6b8e022a2947fbe31e66afc
Torch-mlir: 7190726358ae861ab4c7d23e11e7bf2720421c0b

Build setting.json:
LLVM:
{
    "cmake.sourceDirectory": "/home/qzhuang/code/llvm-project/llvm",
    "cmake.configureArgs": [
        "-DLLVM_ENABLE_PROJECTS=mlir;clang;lld",
        "-DLLVM_BUILD_EXAMPLES=ON",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_TARGETS_TO_BUILD=Native",
        "-DLLVM_USE_SPLIT_DWARF=ON",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_LINKER_TYPE=LLD"                
    ],
    "cmake.buildDirectory": "${workspaceFolder}/build/${buildType}"
}


Torch-MLIR:
{
    "cmake.configureArgs": [
        "-DPython3_FIND_VIRTUALENV=ONLY",
        "-DPython_FIND_VIRTUALENV=ONLY",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DMLIR_DIR=/home/qzhuang/code/llvm-project/build/${buildType}/lib/cmake/mlir/",
        "-DLLVM_DIR=/home/qzhuang/code/llvm-project/build/${buildType}/lib/cmake/llvm/",
        "-DCMAKE_C_COMPILER=/home/qzhuang/code/llvm-project/build/Release/bin/clang",
        "-DCMAKE_CXX_COMPILER=/home/qzhuang/code/llvm-project/build/Release/bin/clang++",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_LINKER_TYPE=LLD"
    ],
    "cmake.buildDirectory": "${workspaceFolder}/build/${buildType}"
}


MLIR-TEST:
{
    "cmake.configureArgs": [
        "-DMLIR_DIR=/home/qzhuang/code/llvm-project/build/${buildType}/lib/cmake/mlir/",
        "-DLLVM_DIR=/home/qzhuang/code/llvm-project/build/${buildType}/lib/cmake/llvm/",
        "-DCMAKE_C_COMPILER=/home/qzhuang/code/llvm-project/build/Release/bin/clang",
        "-DCMAKE_CXX_COMPILER=/home/qzhuang/code/llvm-project/build/Release/bin/clang++",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_LINKER_TYPE=LLD"     
    ],
    "cmake.buildDirectory": "${workspaceFolder}/build/${buildType}",
    "files.associations": {
        "ostream": "cpp"
    }
}


Test env:
export PYTHONPATH=`pwd`/build/Debug/python_packages/torch_mlir:`pwd`/test/python/fx_importer:$PYTHON_PATH
python test/python/fx_importer/basic_test.py


export PYTHONPATH=`pwd`/build/Release/python_packages/torch_mlir:`pwd`/test/python/fx_importer:$PYTHON_PATH
python test/python/fx_importer/basic_test.py


export PATH=/home/qzhuang/code/llvm-project/build/Debug/bin:$PATH 
