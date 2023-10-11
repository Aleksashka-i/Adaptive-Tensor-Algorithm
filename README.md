# Adaptive tensor algorithms for the adaptation of convolution filters.

## Pipeline (Pipeline.ipynb):
### 0. Validate pretrained
Take the pre-trained ResNet32 (``pretrained_models/resnet32-d509ac18.th``). Let's see the accuracy of the original model
```
%run stage_0/validate_pretrained.py

Test: [0/79]	Time 9.200 (9.200)	Loss 0.2115 (0.2115)	Prec@1 93.750 (93.750)
Test: [50/79]	Time 0.332 (0.538)	Loss 0.4470 (0.3828)	Prec@1 90.625 (92.371)
 * Prec@1 92.630
 * Timec@1 0.493
```

### 1. Get initial weights
Take the pre-trained ResNet32 (``pretrained_models/resnet32-d509ac18.th``). Pull out the weights, put the tensors in ``weights/weigths_base.mat``.
```
%run stage_1/get_initial_weights.py
```
Then we decompose the weights in CP-decomposition, using NLS (non-linear least squares), using ``cpd_nls`` from [Tensorlab](https://www.tensorlab.net) in MATLAB (sample script in ``MATLAB/script_matlab_decompose.m``). In the file ``weights/weights_nls_nls.mat``.

### 2. Fine-tune initial decomposition
Replace the set of layers in the model with decomposed (all 3x3 filters) (``weigths/weigths_nls.mat``) using ``cpd_nls``. Then we do filetuning of the weights (``epochs=50``). The best model is saved in ``decomposed/best_initial_decompose.th``. The replaced layers can be seen in ``functional.py``. There are 6 in the example.
```
%run stage_2/fine_tune_initial_decomposition.py
```
![initial_decomposition](stage_2/initial_decomposition.png)

You can run ``stage_2/check.py`` to look at the model structure and accuracy on the test sample y ``decompose/best_initial_decompose.th`` (it's fast). Worse by 1.1 from the original model.
```
%run stage_2/check.py

Test: [0/79]	Time 7.342 (7.342)	Loss 0.3062 (0.3062)	Prec@1 92.969 (92.969)
Test: [50/79]	Time 0.303 (0.508)	Loss 0.5858 (0.4408)	Prec@1 90.625 (91.330)
 * Prec@1 91.520
 * Timec@1 0.430
```
<details>
   <summary><b>%run stage_2/check.py</b></summary>
   <pre>
DataParallel(
  (module): ResNet(
    (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (1): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(8, 8, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=8, bias=False)
          (2_decomposed): Conv2d(8, 8, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=8, bias=False)
          (3_decomposed): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (2): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (3): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(8, 8, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=8, bias=False)
          (2_decomposed): Conv2d(8, 8, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=8, bias=False)
          (3_decomposed): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (4): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): LambdaLayer()
      )
      (1): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(16, 16, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=16, bias=False)
          (2_decomposed): Conv2d(16, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=16, bias=False)
          (3_decomposed): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (2): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (3): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(16, 16, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=16, bias=False)
          (2_decomposed): Conv2d(16, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=16, bias=False)
          (3_decomposed): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (4): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): LambdaLayer()
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=32, bias=False)
          (2_decomposed): Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=32, bias=False)
          (3_decomposed): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (2): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (3): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(32, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=32, bias=False)
          (2_decomposed): Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=32, bias=False)
          (3_decomposed): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (4): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
    )
    (linear): Linear(in_features=64, out_features=10, bias=True)
  )
)
Files already downloaded and verified
Test: [0/79]	Time 7.342 (7.342)	Loss 0.3062 (0.3062)	Prec@1 92.969 (92.969)
Test: [50/79]	Time 0.303 (0.508)	Loss 0.5858 (0.4408)	Prec@1 90.625 (91.330)
 * Prec@1 91.520
 * Timec@1 0.430
   </pre>
</details>

### 3. Extend Decomposed Kernels
We take the defintune model from the previous step (``decomposed/best_initital_decomposed.th``). We enlarge the factors (``1_decomposed``, ``2_decomposed``) to 1x21 and 21x1. We add sigmas ``resnet_with_sigmas.py`` and train with the sigmas. The best approximation model is saved in ``decomposed/best_extended_decomposed.th``.

```
%run stage_3/extend_decomposed_kernels.py
```
![best_extended_decomposition](stage_3/extended_kernels_decomposition.png)
![sigmas](stage_3/sigmas_values.png)

You can run ``stage_3/check.py`` to look at the model structure, sigma values, their corresponding kernel sizes, and accuracy on the test sample y ``decompose/best_extended_decompose.th`` (it's fast). 
```
%run stage_3/check.py

...
best sigmas:  [-1.0235847234725952, -1.6244574785232544, -0.6324220895767212, -1.0306727886199951, -0.5999158024787903, -0.45050451159477234]
best kernels_sz:  [9, 7, 11, 9, 11, 11]
Files already downloaded and verified
Test: [0/79]	Time 7.086 (7.086)	Loss 0.2054 (0.2054)	Prec@1 91.406 (91.406)
Test: [50/79]	Time 0.432 (0.575)	Loss 0.4664 (0.3881)	Prec@1 85.938 (88.817)
 * Prec@1 89.050
 * Timec@1 0.520
```
<details>
   <summary><b>%run stage_3/check.py</b></summary>
   <pre>
         DataParallel(
  (module): ResNet(
    (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (1): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(8, 8, kernel_size=(21, 1), stride=(1, 1), padding=(10, 0), groups=8, bias=False)
          (2_decomposed): Conv2d(8, 8, kernel_size=(1, 21), stride=(1, 1), padding=(0, 10), groups=8, bias=False)
          (3_decomposed): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (2): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (3): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(8, 8, kernel_size=(21, 1), stride=(1, 1), padding=(10, 0), groups=8, bias=False)
          (2_decomposed): Conv2d(8, 8, kernel_size=(1, 21), stride=(1, 1), padding=(0, 10), groups=8, bias=False)
          (3_decomposed): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (4): BasicBlock(
        (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): LambdaLayer()
      )
      (1): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(16, 16, kernel_size=(21, 1), stride=(1, 1), padding=(10, 0), groups=16, bias=False)
          (2_decomposed): Conv2d(16, 16, kernel_size=(1, 21), stride=(1, 1), padding=(0, 10), groups=16, bias=False)
          (3_decomposed): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (2): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (3): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(16, 16, kernel_size=(21, 1), stride=(1, 1), padding=(10, 0), groups=16, bias=False)
          (2_decomposed): Conv2d(16, 16, kernel_size=(1, 21), stride=(1, 1), padding=(0, 10), groups=16, bias=False)
          (3_decomposed): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (4): BasicBlock(
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): LambdaLayer()
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(32, 32, kernel_size=(21, 1), stride=(1, 1), padding=(10, 0), groups=32, bias=False)
          (2_decomposed): Conv2d(32, 32, kernel_size=(1, 21), stride=(1, 1), padding=(0, 10), groups=32, bias=False)
          (3_decomposed): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (2): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (3): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Sequential(
          (0_decomposed): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1_decomposed): Conv2d(32, 32, kernel_size=(21, 1), stride=(1, 1), padding=(10, 0), groups=32, bias=False)
          (2_decomposed): Conv2d(32, 32, kernel_size=(1, 21), stride=(1, 1), padding=(0, 10), groups=32, bias=False)
          (3_decomposed): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
      (4): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (shortcut): Sequential()
      )
    )
    (linear): Linear(in_features=64, out_features=10, bias=True)
  )
)
best sigmas:  [-1.0235847234725952, -1.6244574785232544, -0.6324220895767212, -1.0306727886199951, -0.5999158024787903, -0.45050451159477234]
best kernels_sz:  [9, 7, 11, 9, 11, 11]
Files already downloaded and verified
Test: [0/79]	Time 7.086 (7.086)	Loss 0.2054 (0.2054)	Prec@1 91.406 (91.406)
Test: [50/79]	Time 0.432 (0.575)	Loss 0.4664 (0.3881)	Prec@1 85.938 (88.817)
 * Prec@1 89.050
 * Timec@1 0.520
   </pre>
</details>

### 4. Get decomposed weights
We take the pre-trained model obtained in the previous step, apply a mask to the weights, and crop them to the accepted dimensions. Then we put the decomposed weights into ``weights/decomposed_weights.mat``.
```
%run stage_4/get_decomposed_weights.py

kernels_sz:  [9, 7, 11, 9, 11, 11]
sigmas:  [-1.0235847234725952, -1.6244574785232544, -0.6324220895767212, -1.0306727886199951, -0.5999158024787903, -0.45050451159477234]
```
Then, back composite the tensors using ``cpdgen`` from [Tensorlab](https://www.tensorlab.net) in MATLAB (sample script in ``MATLAB/script_matlab_compose.m``). In the file ``weights/weights_composed.mat``.

### 5. Final fine-tune
Replace the set of layers in the model with normal ``conv2`` layers with new filter sizes and weights compiled in the previous step. We do a final filetuning (``epochs=50``). The best model is saved in ``decomposed/best_final.th``.
```
%run stage_5/final_fine_tune.py
```
![final](stage_5/final_1.png)
You can run ``stage_5/check.py`` to look at the model structure and accuracy on the test sample y ``decomposed/best_final.th`` (this is fa
```
%run stage_5/check.py

Files already downloaded and verified
Test: [0/79]	Time 5.046 (5.046)	Loss 0.1370 (0.1370)	Prec@1 94.531 (94.531)
Test: [50/79]	Time 0.328 (0.419)	Loss 0.3376 (0.2745)	Prec@1 90.625 (91.452)
 * Prec@1 91.550
 * Timec@1 0.383
```
