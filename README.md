# RT-HDIST: Ray-Tracing Core-based Hausdorff Distance Computation  
<img width="2795" height="972" alt="teaser2" src="https://github.com/user-attachments/assets/837b0084-bf68-4514-ad93-9adb48cda953" />
This repository contains the implementation of the following paper: "RT-HDIST: Ray-Tracing Core-based Hausdorff Distance Computation" (Pacific Graphics 2025).   

  
> **Abstract**  
> The Hausdorff distance is a fundamental metric with widespread applications across various fields. However, its computation remains computationally expensive, especially for large-scale datasets. This work targets exact point-to-point Hausdorff distance on point sets. In this work, we present RT-HDIST, the first Hausdorff distance algorithm accelerated by ray-tracing cores (RT-cores). By reformulating the Hausdorff distance problem as a series of nearest-neighbor searches and introducing a novel quantized voxel-index space, RT-HDIST achieves significant reductions in computational overhead while maintaining exact results. Extensive benchmarks demonstrate up to a two-order-of-magnitude speedup over prior state-of-the-art methods, underscoring RT-HDIST's potential for real-time and large-scale applications. 

## Installation  
### CMAKE
```

```
### Dependencies
```
CUDA 12.3 (or higher)
OptiX 7.4.0
```

## Usage
We use the .ptx shader for the RT executable; therefore, the ptx code must be contained in the execution file folder.
```
Release
└ RTHDdemo.exe
└ __shader__hd__qcluster__.ptx
```
If necessary, you can recompile the ptx file using nvcc.  
For detailed implementations, you can refers the code of "shader/__shader__hd__qcluster__.cu" and "demo/demo1/demo.cpp"
```
nvcc.exe --machine=64 --ptx --generate-code arch=compute_75,code=sm_75 --use_fast_math --relocatable-device-code=true --generate-line-info -Wno-deprecated-gpu-targets __shader__hd__qcluster__.cu -ccbin "ccbin folder" -o __shader__hd__qcluster__.ptx -I"OptixPath" -I"libs" -I"libs/3rdParty"
```

### Using Demo
```
RTHDdemo.exe -path 2 "Dataset/ObjectA.obj" "Dataset/ObjectB.obj" -grid 8 8 -tr 0.5 -log "verbose.csv"  
```

<!--
퍼블리쉬 이후 수정할 것
-->
## Citation  
```
@article{kim2025rt,
  title={RT-HDIST: Ray-Tracing Core-based Hausdorff Distance Computation},
  author={Kim, YoungWoo and Lee, Jaehong and Kim, Duksu},
  journal={arXiv preprint arXiv:2504.13436},
  year={2025}
}
```
