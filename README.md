Got it. Here's an updated version of the `README.md` with the necessary changes to use `parameters.xml` instead of JSON:

---

# Customizable Neural Network in Java

This is a simple **Artificial Neural Network** implementation in Java that can be customized according to user needs. The main usage is to approximate functions and visualize the results.

## Overview

This project allows you to create and train a neural network with customizable parameters. You can adjust the network's structure, activation functions, and training parameters to fit your specific requirements.

![Figure 1](https://github.com/user-attachments/assets/7c5de76f-c3b3-4dca-bf8b-8699043f54ed)
![Figure 2](https://github.com/user-attachments/assets/3066af3d-bcc4-4db1-8036-2361e457182c)

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
   pip install -r requirements.txt
   ```

6. **Compile Java Files**

   ```sh
   javac -cp .;C:\libs\json-20210307.jar -d JAVA/OUT JAVA/*.java
   ```

7. **Run the Java Application**

   ```sh
   java -cp .;C:\libs\json-20210307.jar;JAVA/OUT JAVA.Main path/to/parameters.xml
   ```

### Customizing Parameters

You can customize the neural network's parameters by editing the `parameters.xml` file. Experiment with different values to see how they affect the network's performance and output. Here is an example of the `parameters.xml` file:

```xml
<parameters>
    <feature>1</feature>
    <output>1</output>
    <train>3000</train>
    <test>600</test>
    <validate>300</validate>
    <lower_bound>-2</lower_bound>
    <upper_bound>20</upper_bound>
    <structure>1,30,30,30,30,1</structure>
    <functions>linear,reLU,reLU,reLU,reLU,linear</functions>
    <type>Regression</type>
    <epochs>1000</epochs>
    <lr>0.01</lr>
    <Model>adam</Model>
    <beta1>0.99</beta1>
    <beta2>0.999</beta2>
    <epsilon>1e-8</epsilon>
    <batchSize>3000</batchSize>
    <lambda>1e-18</lambda>
</parameters>
```

### Optimizers

- **Gradient Descent**: A basic optimizer that updates weights incrementally in the direction of the negative gradient.
- **Momentum**: Accelerates gradient descent by considering the previous update direction.
- **RMSprop**: Adapts the learning rate for each parameter by dividing by the moving average of squared gradients.
- **Adam**: Combines the advantages of RMSprop and Momentum, providing an adaptive learning rate for each parameter.

### Initializers

- **He**: Initializes weights to values drawn from a truncated normal distribution centered on 0 with a standard deviation of `sqrt(2/fan_in)`.
- **Xavier**: Initializes weights to values drawn from a truncated normal distribution centered on 0 with a standard deviation of `sqrt(1/fan_avg)`.

### Activation Functions

- **Linear**: The identity function.
- **ReLU**: Rectified Linear Unit function, which outputs the input directly if it is positive, otherwise, it outputs zero.
- **Sigmoid**: A logistic function that outputs values between 0 and 1.
- **Tanh**: The hyperbolic tangent function, which outputs values between -1 and 1.

### Example Workflow

1. Clone the repository and navigate to the project directory.
2. Set up and activate the Python virtual environment.
3. Install the required Python packages.
4. Compile the Java files with the necessary dependencies.
5. Run the Java application with the path to your `parameters.xml` file.
6. Modify the parameters in `parameters.xml` to explore different configurations.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to adjust the URLs and repository details according to your actual project. This `README.md` provides a clear and attractive presentation of your project, making it easier for others to understand and use.
