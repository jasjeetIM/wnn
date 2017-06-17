# wnn
**Wide Neural Network**(MNIST Digit Classification):

This is a prototype implementation of a Wide Neural Network. Details on the architecture will be added soon. <br />
The primary architectural difference is that connections from every lower layer go to every higher layer. <br />
Further, the final prediction is made only after neurons in all layers have fired. This set up makes gradient calculations and <br />
backpropagation more complex. <br /> <br />



The network is trained on the MNIST digit classification task. <br />
The modules are written in Python and only use Numpy. <br /> <br />

**Completed**: <br />
Network initialization<br />
Network Forward <br />
Evaluate gradients - weights, v_in, and v_board <br /> 
Softmax <br />
Softmax Loss and derivative <br />
Vectorization  <br />
Complete gradient implementation for biases <br />
Update parameters with gradient <br />
Numerical gradient checker <br />
Test network gradients <br />


**To Do**: <br />
Train on mnist digit classification (in progress) <br />
