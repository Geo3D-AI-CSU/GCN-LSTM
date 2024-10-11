from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

# Import custom utility functions
from spektral_utilities import filter_dot, dot, localpooling_filter


# Define the GraphConvolution layer class
class GraphConv(Layer):
    """
    A graph convolutional layer (GCN) as presented by
    [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).
    **Mode**: single, mixed, batch.
    This layer computes:
    $$
        \Z = \hat \D^{-1/2} \hat \A \hat \D^{-1/2} \X \W + \b
    $$
    where \( \hat \A = \A + \I \) is the adjacency matrix with added self-loops
    and \(\hat\D\) is its degree matrix.

    **Input**
    - Node features of shape `([batch], N, F)`;
    - Modified Laplacian of shape `([batch], N, N)`; can be computed with
    `spektral.utils.convolution.localpooling_filter`.

    **Output**
    - Node features with the same shape as the input, but with the last
    dimension changed to `channels`.

    **Arguments**
    - `channels`: number of output channels;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    """

    def __init__(self,
                 channels,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        # Call the constructor of the parent class, setting up activity regularization
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Number of output channels (i.e., the number of features after graph convolution)
        self.channels = channels

        # Activation function
        self.activation = activations.get(activation)

        # Whether to use bias in the linear transformation
        self.use_bias = use_bias

        # Initializer for the kernel matrix
        self.kernel_initializer = initializers.get(kernel_initializer)

        # Initializer for the bias vector
        self.bias_initializer = initializers.get(bias_initializer)

        # Regularization applied to the kernel matrix
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Regularization applied to the bias vector
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Constraint applied to the kernel matrix
        self.kernel_constraint = constraints.get(kernel_constraint)

        # Constraint applied to the bias vector
        self.bias_constraint = constraints.get(bias_constraint)

        # Whether the layer supports masking
        self.supports_masking = False

    # Build the layer, initializing the weights
    def build(self, input_shape):
        assert len(input_shape) >= 2

        # The last dimension of the input is the dimensionality of the node features
        input_dim = input_shape[0][-1]

        # Add the weight matrix and initialize it
        self.kernel = self.add_weight(shape=(input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # If using bias, add and initialize it
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Mark the layer as built
        self.built = True

    # Forward pass
    def call(self, inputs):
        # Get the input node features and adjacency matrix
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = dot(features, self.kernel)
        output = filter_dot(fltr, output)

        # Add bias if applicable
        if self.use_bias:
            output = K.bias_add(output, self.bias)

        # Apply activation function
        if self.activation is not None:
            output = self.activation(output)

        return output

    # Compute the output shape
    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape

    # Get configuration information
    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        # Get configuration from the parent class
        base_config = super().get_config()

        # Return the merged configuration
        return dict(list(base_config.items()) + list(config.items()))

    # Preprocess the adjacency matrix
    @staticmethod
    def preprocess(A):
        return localpooling_filter(A)
