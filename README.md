#                  cuTensor CUDA Library for Tensors

We implement the cuTensor library for the transform-based tensor model.  
>    | CuOperations | tfft | tproduct | tsvd | tqr | tinv | tnorm |
>    | :---: | :---:| :---:| :---: | :---: | :---: | :---: |
>    | based | √ | √ | √ | √ | √ | √ |
>    | streamed | √ | √ | √ | √ |  √ | √ |
>    | batched | √ | √ | √ | √ | √ | √ |

[1] X.-Y. Liu and X. Wang. Fourth-order tensor space with two-dimensional discrete transforms. arXiv,2017.  


## INSTALLION

**1.get CuTensor library from git repository**  
```   
    $ git clone https://github.com/hust512/cuTensor_CUDA_Library_for_Tensors
```
**2.go into cuTensor_CUDA_Library_for_Tensors folder**  
```
    $ cd cuTensor_CUDA_Library_for_Tensors
```
**3.edit make.inc file**  
```
     provide path for third party libraries.you can refer to make.inc_example.  
```
**4.make cuTensor_CUDA_Library_for_Tensors**  
```
    $ make
```
## RESULT
<div style="float:left"><img width="300" height="300" src="https://github.com/lihailihai/Tensor_/blob/master/RESULT/tprod13.png"/></div>
<div style="float:left"><img width="300" height="300" src="https://github.com/lihailihai/Tensor_/blob/master/RESULT/tsvd5.png"/></div>

## CONTACT US

> **Tel:** \+86 21 6613 5300  
> **E-Mail:** taozhang@shu.edu.cn  
> **IIPLab:**[http;//www.findai.net](http://www.findai.net)  
> **TensorLet Group: [http://www.tensorlet.com] 
> **github:**[https://github.com/hust512/cuTensor_CUDA_Library_for_Tensors](https://github.com/hust512/cuTensor_CUDA_Library_for_Tensors)  
