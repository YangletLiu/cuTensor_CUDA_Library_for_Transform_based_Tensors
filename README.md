#                  cuTensor CUDA Library for Tensors

We implement the cuTensor library for the transform-based tensor model.  
>    | CuOperations | tfft | tproduct | tsvd | tqr | tinv | tnorm |
>    | :---: | :---:| :---:| :---: | :---: | :---: | :---: |
>    | based | √ | √ | √ | √ | √ | √ |
>    | streamed | √ | √ | √ | √ |  √ | √ |
>    | batched | √ | √ | √ | √ | √ | √ |

[1] T. Zhang, X.-Y. Liu, X. Wang, and A. Walid. cutensor-tubal: Efficient primitives for tubal-rank tensor operations on GPUs. IEEE Transactions on Parallel and Distributed Systems, 2019.

[2] T. Zhang, X.-Y. Liu, and X. Wang. High Performance GPU tensor completion with tubal-sampling pattern. IEEE Transactions on Parallel and Distributed Systems, 2020.


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

> Tensor and Deep Learning Lab LLC. 

> E-Mail:** tensorlet@gmail.com  

> TensorLet Group: [http://www.tensorlet.com/projects/cutensor/] 

> github:**[https://github.com/hust512/cuTensor_CUDA_Library_for_Tensors](https://github.com/hust512/cuTensor_CUDA_Library_for_Tensors)  

> Notice: both US and China patents are filed for this project.
