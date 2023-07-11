# 🎸 🤘 rocBLAS Tune Everything Rapidly 🤷
## Build
`pip3 install -r requirements.txt`<br>
`pip3 install -v .`
## Use
`rocBlaster ${YOUR COMMAND HERE}`
Example:
`rocBlaster python3 micro_benchmarking_pytorch.py --network alexnet --iterations 1 --fp16 1`
## TODO:
- Support Batched and Stridded
- Support different sizes (fp32...)
- Support Multi-GPU parallelism
- ...
