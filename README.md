# Personalized Algorithm Generation: A Case Study in Learning ODE Integrators

## Requirements
This project uses `python 3.8`. Set up the project using the following step:

Install 
```shell
$pip install -r requirements.txt
```


## Test on Different Targets
Look at [examples](./examples).

* Go to the folder
    ```shell
    $cd examples
    ```
* Run the training process (eg. linear traget)
    ```shell
    $python main_linear.py
    ```
* Change the parameters (such as learning rate, integrator stage, targeted order and so on) in [configuration file](./examples/config.yml)
    * For "linear target", a huge weight is multiplied on Taylor-based regularizer as a scale because this value is too small.
    * For "square_nonlinear target", we multiply an increasing weight on MSE loss and a decreasing one on Taylor-based regularizer in order to focus on different loss in the training.


## See the evaluation 
* Ordercheck
    Run ipynb file [ordercheck](./ordercheck.ipynb).
* Plot the MSE on Van der Pol Oscillator and the Brussealtor.
    Run ipynb file [plot error bar](./plot_error_bar.ipynb).
* See the figures according to above evaluation in [results](./results).
