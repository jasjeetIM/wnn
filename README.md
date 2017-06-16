# wnn
Wide Neural Network:

This is a prototype implementation of a Wide Neural Network. Details on the architecture will be added soon. <br />
The primary architectural difference is that connections from every lower layer go to every higher layer. <br />
Further, the final prediction is made only after neurons in all layers have fired. This set up makes gradient calculations and <br />
backpropagation more complex. <br />
The modules are written in Python and only use Numpy. <br /> <br />

Completed: <br />
Network initialization<br />
Network Forward <br />
Evaluate gradients - weights and v_in, v_board <br /> 
Softmax <br />
Softmax Loss and derivative <br />
Vectorization over batch <br />
Complete gradient implementation for biases <br />
Update parameters with gradient <br />

To Do: <br />
Unit testing <br />
Train on mnist digit classification <br />
