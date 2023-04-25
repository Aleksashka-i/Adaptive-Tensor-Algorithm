# CourseProject
Adaptive tensor algorithms for the adaptation of convolution filters.


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
   Test: [0/79]	Time 4.911 (4.911)	Loss 0.3217 (0.3217)	Prec@1 90.625 (90.625)
   Test: [50/79]	Time 0.264 (0.353)	Loss 0.6284 (0.5401)	Prec@1 89.062 (88.496)
    * Prec@1 88.740
    * Timec@1 0.317
   ```
   
   After fine-tuning:
   ```
   Test: [0/79]	Time 5.518 (5.518)	Loss 0.4164 (0.4164)	Prec@1 90.625 (90.625)
   Test: [50/79]	Time 0.260 (0.366)	Loss 0.5170 (0.5306)	Prec@1 87.500 (89.354)
    * Prec@1 89.510
    * Timec@1 0.327
   ```
