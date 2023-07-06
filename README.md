pip install pybind11[global]
```
mkdir -p build
cd build
CXX=/opt/rocm/bin/hipcc cmake ..
make
```
