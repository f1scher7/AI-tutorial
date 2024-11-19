# AI-tutorial
AI tutorial

## Hidden Layer Types
| **Hidden Layer Type**     | **Application**                                    | **When to Use**                                                              | **How They Differ**                                                             |
|---------------------------|----------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **Dense**                 | General tasks in classification and regression.    | Use for general-purpose tasks, especially when all input features interact with each other. Typically used in feedforward neural networks. | All neurons are connected to each other. Used for general tasks where every input influences every output. |
| **Convolutional**         | Image and spatial data analysis.                  | Use for image data, video, or any task where local spatial relationships need to be captured (e.g., image classification, object detection). | Uses filters to capture local patterns in data. Each neuron is connected to a local region of the input, rather than the entire input. |
| **Recurrent (RNN, LSTM)** | Sequential data (e.g., text, time series).         | Use for sequential data or time-series problems, where past information influences the present (e.g., text generation, time-series forecasting). | Neurons have connections to themselves, enabling them to process sequences and maintain memory over time. |
| **Dropout**               | Protection against overfitting.                   | Use when you have a deep neural network and want to prevent overfitting by randomly deactivating neurons during training. | Randomly disables a percentage of neurons in each training iteration to prevent overfitting. |
| **Batch Normalization**   | Accelerating training.                            | Use when training deep networks, as it normalizes intermediate layers and improves convergence speed, stability, and generalization. | Normalizes the activations of neurons within a minibatch to improve training stability and speed. |
| **Pooling**               | Dimensionality reduction in image analysis.       | Use to reduce the spatial dimensions of data, typically in CNNs, to decrease computation and prevent overfitting. | Reduces the size of the data by summarizing regions of the input (e.g., max pooling or average pooling). |
| **Residual**              | Very deep networks.                               | Use in very deep networks (like ResNet) to prevent vanishing gradients and to allow for better training in deep architectures. | Introduces shortcut connections that skip one or more layers to allow gradients to flow more easily through deep networks. |

## Normalization Types
| **Normalization Type**        | **Application**                                                                 | **When to Use**                                      |
|-------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------|
| **Min-Max Scaling**            | Scales data to a fixed range (e.g., 0 to 1).                                    | Use when data has known min and max values; ideal for algorithms that require normalized data within a specific range (e.g., neural networks, k-NN). |
| **Standardization (Z-score)**  | Centers data by subtracting the mean and scaling by the standard deviation.     | Use when data follows a Gaussian distribution; ideal for models that assume normally distributed data (e.g., linear regression, logistic regression, SVM). |
| **Robust Scaling**             | Uses the median and interquartile range (IQR) to scale, robust to outliers.     | Use when data contains significant outliers or skewed distributions; ideal for models that are sensitive to outliers (e.g., tree-based models). |
| **MaxAbs Scaling**             | Scales data to the range [-1, 1] using the maximum absolute value.              | Use when data is already centered around zero and you want to preserve the sparsity of the data (e.g., sparse matrices in NLP). |
| **Quantile Transformation**    | Maps data to a uniform or normal distribution using quantiles.                  | Use when you need to map data to a specific distribution (e.g., for Gaussian-based models or when normalizing skewed data). |
| **Log Transformation**         | Reduces the impact of outliers by applying the logarithm.                        | Use when data is highly skewed or contains extreme values (e.g., for financial data or data with exponential growth patterns). |
| **Power Transformation**       | Applies a power function (e.g., Box-Cox, Yeo-Johnson) to stabilize variance and normalize data. | Use when data is skewed and you want to stabilize variance, especially for data that doesn't fit a normal distribution. |

## Initialization Types
| **Activation Function** | **Recommended Initialization**            | **Reason**                                               |
|-------------------------|-------------------------------------------|---------------------------------------------------------|
| **Sigmoid, Tanh**       | Xavier Initialization                    | Balances gradients for symmetric activations.           |
| **ReLU, Leaky ReLU**    | He Initialization                        | Accounts for ReLU's sparse activations.                 |
| **Linear**              | Xavier Initialization                    | Preserves variance for linear activations.              |
| **Swish, GELU**         | He Initialization or Variance Scaling    | Similar to ReLU; focuses on handling non-linearity.     |

## Random and Normal Initialization
| **Feature**               | **Random Initialization**                        | **Normal Initialization**                             |
|---------------------------|--------------------------------------------------|-------------------------------------------------------|
| **Distribution Type**      | Uniform \([0, 1)\)                              | Normal (Gaussian) mu = 0, sigma = 1                   |
| **Weight Range**           | Only positive values \([0, 1)\)                 | Both positive and negative values                     |
| **Value Distribution**     | Uniform probability                              | Concentrated around 0, fewer extreme values           |
| **Application**            | Simple networks (few layers)                    | More complex neural networks                          |
| **Advantages**             | Easy to implement                               | Weights more suitable for deep networks               |

## Xavier and He Initialization
| **Feature**               | **Xavier Initialization**                                 | **He Initialization**                           |
|---------------------------|-----------------------------------------------------------|-------------------------------------------------|
| **Purpose**               | Symmetric activations (e.g., Tanh, Sigmoid)               | Asymmetric activations (e.g., ReLU, Leaky ReLU) |
| **Variance Calculation**  | Dependent on Nin and Nout: | Dependent only on Nin                           |
| **Value Range**           | Smaller (more concentrated around zero)                   | Larger, adapted for ReLU                        |
| **Advantages**            | Balanced signal propagation                               | Tailored for ReLU activation                    |
| **Disadvantages**         | May be too weak for ReLU                                  | Ineffective for Sigmoid and Tanh                |
