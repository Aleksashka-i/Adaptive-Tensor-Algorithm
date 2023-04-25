# CourseProject
### Adaptive tensor algorithms for the adaptation of convolution filters.


**Examples:**

1. Validation of [pretrained](https://github.com/akamaster/pytorch_resnet_cifar10) ResNet-32:
   ```
   python3 validate_pretrained.py
   ```
   ```
   Test: [0/79]	Time 4.095 (4.095)	Loss 0.2115 (0.2115)	Prec@1 93.750 (93.750)
   Test: [50/79]	Time 0.262 (0.333)	Loss 0.4470 (0.3828)	Prec@1 90.625 (92.371)
    * Prec@1 92.630
    * Timec@1 0.304
   ```

2. Replace on of the convolutional layers with [CP-decomposition](https://arxiv.org/pdf/1412.6553.pdf):
   ```
   python3 cp_decomposition.py
   ```
   <pre>
   (0): BasicBlock(
     (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     <b>(conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)</b>
     (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (shortcut): LambdaLayer()
   )
   
   <b>Total Trainable Params: 464154</b>
   
   ------------------>
   
   (0): BasicBlock(
     (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     <b>(conv2): Sequential(
       (0_decomposed): Conv2d(64, 21, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (1_decomposed): Conv2d(21, 21, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=21, bias=False)
       (2_decomposed): Conv2d(21, 21, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), groups=21, bias=False)
       (3_decomposed): Conv2d(21, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
     )</b>
     (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (shortcut): LambdaLayer()
   )
   
   <b>Total Trainable Params: 430104</b>
   </pre>

3. Fine-tune entire network after replacement (epochs = 1):

   Before fine-tuning:
   ```
   Test: [0/79]	Time 4.866 (4.866)	Loss 0.2535 (0.2535)	Prec@1 92.188 (92.188)
   Test: [50/79]	Time 0.259 (0.352)	Loss 0.5016 (0.4857)	Prec@1 89.062 (89.507)
    * Prec@1 89.770
    * Timec@1 0.317
   ```
   
   After fine-tuning:
   ```
   Test: [0/79]	Time 5.269 (5.269)	Loss 0.1993 (0.1993)	Prec@1 91.406 (91.406)
   Test: [50/79]	Time 0.269 (0.362)	Loss 0.4391 (0.4097)	Prec@1 89.844 (90.380)
    * Prec@1 90.520
    * Timec@1 0.324
   ```
