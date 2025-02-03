# Customizable Neural Network in Java

This is a simple **Artificial Neural Network** implementation in Java that can be customized according to user needs. The main usage is to approximate functions and visualize the results.

## Overview

This project allows you to create and train a neural network with customizable parameters. You can adjust the network's structure, activation functions, and training parameters to fit your specific requirements.

## Snapshots

![ACTUAL](https://github.com/user-attachments/assets/ee3cab14-711f-4640-a673-d173e6da3a71)
![PREDICTED](https://github.com/user-attachments/assets/6795d669-b6cf-4685-bf17-54a4dd5f6b7d)

## Evolution

![evolution-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/84e45827-a26f-4e86-a109-922566086653)


## Features

- **Customizable Neural Network Structure**: Define the number of layers and neurons in each layer.
- **Activation Functions**: Choose from linear, ReLU, sigmoid, and tanh.
- **Optimizers**: Use gradient descent, momentum, RMSprop, or Adam for training.
- **Initializers**: Initialize weights using He or Xavier methods.
- **Visualization**: Visualize the function approximations using Python scripts.

## Usage

### Prerequisites

- Java Development Kit (JDK)
- Python (with `venv` module)

### Steps to Run the Project

1. **Clone the Repository**

   ```sh
   git clone https://github.com/your-repo/Neural-Networks.git
   ```

2. **Navigate to the Project Directory**

   ```sh
   cd Neural-Networks
   ```

3. **Set Up Python Virtual Environment (One-Time Setup)**

   ```sh
   python -m venv PYTHON/venv
   ```

4. **Activate the Python Virtual Environment**

   - On Windows:

     ```sh
     PYTHON\venv\Scripts\activate
     ```

   - On Unix or macOS:

     ```sh
     source PYTHON/venv/bin/activate
     ```

5. **Install Required Python Packages**

   ```sh
   pip install -r PYTHON/requirements.txt
   ```

6. **Compile Java Files**

   ```sh
   javac -d JAVA/OUT JAVA/*.java
   ```

7. **Run the Java Application**

   For visualizing the function
   ```sh
   java -cp JAVA/OUT JAVA/Main --visualize parameters.xml
   ```
   For training the model
   ```sh
   java -cp JAVA/OUT JAVA/Main --run parameters.xml
   ```

### Customizing Parameters

You can customize the neural network's parameters by editing the `parameters.xml` file. Experiment with different values to see how they affect the network's performance and output. Here is an example of the `parameters.xml` file:

```xml
<parameters>
    <!-- Number of input nodes (depends on your specific problem) -->
    <feature>1</feature>
    <!-- Number of output nodes (depends on your specific problem) -->
    <output>1</output>
    <!-- Number of train data (more data generally leads to better performance, usually in the thousands) -->
    <train>3000</train>
    <!-- Number of test data (typically 10-20% of the total dataset) -->
    <test>600</test>
    <!-- Number of validation data (typically 10-20% of the total dataset) -->
    <validate>300</validate>
    <!-- Lower bound range for input generation function (specific to your problem) -->
    <lower_bound>-2</lower_bound>
    <!-- Upper bound range for input generation function (specific to your problem) -->
    <upper_bound>20</upper_bound>
    <!-- Neural network structure [feature, ... hidden layers ... , output] (depends on the complexity of your problem) -->
    <structure>1,30,30,30,30,1</structure>
    <!-- Activation function of each layer (linear, reLU, sigmoid, tanh) (ReLU is commonly used for hidden layers, linear for output in regression) -->
    <functions>linear,reLU,reLU,reLU,reLU,linear</functions>
    <!-- Type of the model (Regression, Classification) -->
    <type>Regression</type>
    <!-- Total number of epochs (typically ranges from 100 to 1000) -->
    <epochs>1000</epochs>
    <!-- Learning rate (commonly ranges from 0.001 to 0.01) -->
    <lr>0.01</lr>
    <!-- Optimizers (gradient_decent, momentum, rms_prop, adam) (Adam is a popular choice) -->
    <Model>adam</Model>
    <!-- Parameter for momentum or adam (commonly 0.9 for momentum; for Adam, beta1 is often set between 0.9 and 0.999) -->
    <beta1>0.99</beta1>
    <!-- Parameter for rms_prop or adam (for Adam, beta2 is typically set between 0.999 and 0.9999) -->
    <beta2>0.999</beta2>
    <!-- Parameter for rms_prop or adam (epsilon is typically set between 1e-7 and 1e-8) -->
    <epsilon>1e-8</epsilon>
    <!-- Training batch size (commonly ranges from 32 to 256, but can be larger for more stable training) -->
    <batchSize>3000</batchSize>
    <!-- Parameter for regularization (commonly ranges from 1e-5 to 1e-3) -->
    <lambda>1e-18</lambda>
    <!-- Switch for evolution video (specific to your preference) -->
    <video>true</video>
</parameters>
```

### Optimizers (*change the **parameters.xml** accordingly*)

- **Gradient Descent**: A basic optimizer that updates weights incrementally in the direction of the negative gradient.
   ```sh
   gradient-descent
   ```
- **Momentum**: Accelerates gradient descent by considering the previous update direction.
   ```sh
   momentum
   ```
- **RMSprop**: Adapts the learning rate for each parameter by dividing by the moving average of squared gradients.
   ```sh
   rms_prop
   ```
- **Adam**: Combines the advantages of RMSprop and Momentum, providing an adaptive learning rate for each parameter.
   ```sh
   adam
   ```

### Initializers (*change the **parameters.xml** accordingly*)

- **He**: Initializes weights to values drawn from a truncated normal distribution centered on 0 with a standard deviation of `sqrt(2/fan_in)`.
   ```sh
   he
   ```
- **Xavier**: Initializes weights to values drawn from a truncated normal distribution centered on 0 with a standard deviation of `sqrt(1/fan_avg)`.
   ```sh
   xavier
   ```

### Activation Functions (*change the **parameters.xml** accordingly*)

- **Linear**: The identity function.
   ```sh
   linear
   ```
- **ReLU**: Rectified Linear Unit function, which outputs the input directly if it is positive, otherwise, it outputs zero.
   ```sh
   reLU
   ```
- **Sigmoid**: A logistic function that outputs values between 0 and 1.
   ```sh
   sigmoid
   ```
- **Tanh**: The hyperbolic tangent function, which outputs values between -1 and 1.
   ```sh
   tanh
   ```

### Example Workflow

1. Clone the repository and navigate to the project directory.
2. Set up and activate the Python virtual environment.
3. Install the required Python packages.
4. Compile the Java files with the necessary dependencies.
5. Run the Java application with the type and path to your `parameters.xml` file.
6. Modify the parameters in `parameters.xml` to explore different configurations.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
