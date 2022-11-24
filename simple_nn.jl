using LinearAlgebra, Distributions, Random, Zygote

struct DenseLayer
    size::Tuple{Int, Int}
    params::Dict
    operation::Function
    activation::Function

    function DenseLayer(size::Tuple{Int, Int}, activation::Function; params=Dict())
        return new(size, params, dense_layer, activation)
    end
end

struct ConvLayer
    size::Tuple{Int, Int, Int}  # in channels, out channels, kernel size
    params::Dict                # padding, stride
    operation::Function
    activation::Function

    function ConvLayer(size::Tuple{Int, Int, Int}, activation::Function; params=Dict("Stride" => 1, "Padding" => 1))
        return new(size, params, conv_2d_layer, activation)
    end
end

struct PoolLayer
    size::Tuple{Int, Int, Int}  # in channels, out channels, kernel size
    params::Dict                # padding, stride
    operation::Function
    pool_function::Function

    function ConvLayer(size::Tuple{Int, Int, Int}, pool_operation::Function; params=Dict("Stride" => 1, "Padding" => 1))
        return new(size, params, pool_2d, pool_operation)
    end
end

mutable struct Network
    layers::Vector
    biases::Vector
    weights::Vector
    params::Dict
end

function CreateNetwork(setup; datatype=Float32, init_distribution=Normal())
    biases, weights = [], []
    for layer in setup
        if typeof(layer) == DenseLayer
            push!(biases, convert.(datatype, rand(init_distribution, layer.size[2])))
            push!(weights, convert.(datatype, rand(init_distribution, layer.size[2], layer.size[1])))

        elseif typeof(layer) == ConvLayer || typeof(layer) == PoolLayer
            push!(biases, [
                convert.(datatype, rand(init_distribution)) for _ in 1:layer.size[2]
            ])

            push!(weights, [
                [convert.(datatype, rand(init_distribution, layer.size[3], layer.size[3])) for _ in 1:layer.size[1]] for _ in 1:layer.size[2]
            ])
        end
    end
    
    return Network(setup, biases, weights, Dict(["datatype" => datatype]))
end

function Forward(net::Network, a)
    for layer in 1:length(net.weights)
        if typeof(net.layers[layer]) == DenseLayer
            if layer > 1 && (typeof(net.layers[layer-1]) == ConvLayer || typeof(net.layers[layer]) == PoolLayer)
                a = vec(vcat(a...))
            end

            a = net.layers[layer].activation(net.layers[layer].operation(net, layer, a))
        
        elseif typeof(net.layers[layer]) == ConvLayer
            a = [net.layers[layer].activation.(mat) for mat in net.layers[layer].operation(net, layer, a)]
        
        elseif typeof(net.layers[layer]) == PoolLayer
            a = net.layers[layer].operation(net, layer, a)
        end
    end
    
    return a
end

function OptimizerSetup!(net::Network, optimizer; optimizer_params...)
    optimizer_params = get_optimizer_params(net, optimizer; optimizer_params...)
    net.params = optimizer_params
end

function Backward!(net::Network, loss_function, x, y)
    weight_grad = gradient(Params([net.weights])) do
        loss_function(x, y)
    end

    bias_grad = gradient(Params([net.biases])) do
        loss_function(x, y)
    end
    
    net.params["optimizer"](net, weight_grad[net.weights], bias_grad[net.biases])
end

# functions
# layer types
function dense_layer(net::Network, layer::Int, a)
    return vec(a' * net.weights[layer]') .+ net.biases[layer]
end

feature_dim(image_dim, kernel_dim, padding, stride) = Int((image_dim - kernel_dim + 2*padding) / stride + 1)

# function pad_matrix(matrix, padding; pad_value=0)
#     padded_matrix = ones(size(matrix) .+ 2*padding) .* pad_value
#     padded_matrix[1+padding:end-padding, 1+padding:end-padding] = matrix

#     return padded_matrix
# end

function pad_matrix(mat, padding; pad_value=0.0)
    n = size(mat, 2)    
    return Matrix(PaddedView(pad_value, mat, (n + 2*padding, n + 2*padding), (1 + padding, 1 + padding)))
end

function conv_2d_layer(net::Network, layer::Int, a)
    conv_layer = net.layers[layer]
    # weights = net.weights[layer]
    # biases = net.biases[layer]
    padding, stride = conv_layer.params["Padding"], conv_layer.params["Stride"]
    
    n_f = feature_dim(size(a[1], 1), size(net.weights[layer][1][1], 1), padding, stride)
    n_k = size(net.weights[layer][1][1], 1)

    output_volume = [[
        sum([
            dot(image[(m + (m-1)*(stride-1)):m + (m-1)*(stride-1)+(n_k-1), (n + (n-1)*(stride-1)):n + (n-1)*(stride-1)+(n_k-1)], net.weights[layer][v][i]) for (i, image) in enumerate(a)
        ]) .+ net.biases[layer][v] for m in 1:n_f, n in 1:n_f
    ] for v in 1:size(net.weights[layer], 1)]
    
    return output_volume
end

function pool_2d(net::Network, layer::Int, a)
    stride = net.layers[layer].params("Stride")

    n_f = feature_dim(size(a[1], 1), length(a), 0, stride)

    output_volume = [
        net.layers[layer].pool_operation.([mat[(m + (m-1)*(stride-1)):m + (m-1)*(stride-1)+(kernel_size-1), (n + (n-1)*(stride-1)):n + (n-1)*(stride-1)+(kernel_size-1)] for m in 1:n_f, n in 1:n_f]...) for mat in a
    ]

    return output_volume
end

# activation functions
function none_activation(z)
    return z
end

function sigmoid_activation(z)
    return 1 ./ (1 .+ exp.(-z))
end

function relu_activation(z)
    return max.(Float32(0.0), z)
end

function softmax_activation(z)
    return exp.(z) ./ sum(exp.(z))
end

# optimizer stuff
function get_optimizer_params(net::Network, optimizer; optimizer_params...)
    if string(optimizer) == "GradientDescentOptimizer!"
        params = Dict([
            "optimizer" => optimizer,
            "optimizer_name" => "GradientDescentOptimizer!",
            "learning_rate" => 0.01
        ])
    elseif string(optimizer) == "MomentumOptimizer!"
        params = Dict([
            "optimizer" => optimizer,
            "optimizer_name" => "MomentumOptimizer!",
            "learning_rate" => 0.01,
            "gamma" => 0.9,
            "weights_momentum_vector" => [zeros(size(layer)) for layer in net.weights],
            "biases_momentum_vector" => [zeros(size(layer)) for layer in net.biases]
        ])
    elseif string(optimizer) == "RMSpropOptimizer!"
        params = Dict([
            "optimizer" => optimizer,
            "optimizer_name" => "RMSpropOptimizer!",
            "learning_rate" => 0.01,
            "moving_average" => 0.9,
            "epsilon" => 1.0e-8,
            "weights_grad_vec" => [zeros(size(layer)) for layer in net.weights],
            "biases_grad_vec" => [zeros(size(layer)) for layer in net.biases]
        ])
    elseif string(optimizer) == "AdamOptimizer!"
        params = Dict([
           "optimizer" => optimizer,
           "optimizer_name" => "AdamOptimizer!",
           "decay_1" => 0.9,
           "decay_2" => 0.999,
           "step_size" => 0.01,
           "epsilon" => 1.0e-8,
           "weights_m" => [zeros(size(layer)) for layer in net.weights],
           "weights_v" => [zeros(size(layer)) for layer in net.weights],
           "biases_m" => [zeros(size(layer)) for layer in net.biases],
           "biases_v" => [zeros(size(layer)) for layer in net.biases]
       ])
    end
    
    if length(optimizer_params) > 0
        for param in optimizer_params
            params[string(param[1])] = param[2]
        end
    end
    
    return params
end

function GradientDescentOptimizer!(net::Network, weight_grad, bias_grad)
    net.weights = net.weights .- net.params["learning_rate"] * weight_grad
    net.biases = net.biases .- net.params["learning_rate"] * bias_grad
end

function MomentumOptimizer!(net::Network, weight_grad, bias_grad)
    net.params["weights_momentum_vector"] = net.params["gamma"] * net.params["weights_momentum_vector"] + net.params["learning_rate"] * weight_grad
    net.weights = net.weights .- net.params["weights_momentum_vector"]
    
    net.params["biases_momentum_vector"] = net.params["gamma"] * net.params["biases_momentum_vector"] + net.params["learning_rate"] * bias_grad
    net.biases = net.biases .- net.params["biases_momentum_vector"]
end

function RMSpropOptimizer!(net::Network, grad) 
    for layer in 1:length(net.layers)
        net.params["weights_grad_vec"][layer] = net.params["moving_average"] .* net.params["weights_grad_vec"][layer] .+ (1 .- net.params["moving_average"]) .* grad[net.weights][layer].^2
        net.params["biases_grad_vec"][layer] = net.params["moving_average"] .* net.params["biases_grad_vec"][layer] .+ (1 .- net.params["moving_average"]) .* grad[net.biases][layer].^2
        
        net.weights[layer] = net.weights[layer] .- net.params["learning_rate"] ./ sqrt.(net.params["weights_grad_vec"][layer] .+ net.params["epsilon"]) .* grad[net.weights][layer]
        net.biases[layer] = net.biases[layer] .- net.params["learning_rate"] ./ sqrt.(net.params["biases_grad_vec"][layer] .+ net.params["epsilon"]) .* grad[net.biases][layer]
    end
end

function AdamOptimizer!(net::Network, grad)
    for layer in 1:length(net.layers)
        net.params["weights_m"][layer] = net.params["decay_1"] .* net.params["weights_m"][layer] .+ (1 .- net.params["decay_1"]) .* grad[net.weights][layer]
        weights_m_hat = net.params["weights_m"][layer] ./ (1 .- net.params["decay_1"])
        
        net.params["weights_v"][layer] = net.params["decay_2"] .* net.params["weights_v"][layer] .+ (1 .- net.params["decay_2"]) .* grad[net.weights][layer].^2
        weights_v_hat = net.params["weights_v"][layer] ./ (1 .- net.params["decay_2"])
        
        net.params["biases_m"][layer] = net.params["decay_1"] .* net.params["biases_m"][layer] .+ (1 .- net.params["decay_1"]) .* grad[net.biases][layer]
        biases_m_hat = net.params["biases_m"][layer] ./ (1 .- net.params["decay_1"])
        
        net.params["biases_v"][layer] = net.params["decay_2"] .* net.params["biases_v"][layer] .+ (1 .- net.params["decay_2"]) .* grad[net.biases][layer].^2
        biases_v_hat = net.params["biases_v"][layer] ./ (1 .- net.params["decay_2"])
        
        net.weights[layer] = net.weights[layer] .- weights_m_hat .* (net.params["step_size"] ./ (sqrt.(weights_v_hat) .+ net.params["epsilon"]))
        net.biases[layer] = net.biases[layer] .- biases_m_hat .* (net.params["step_size"] ./ (sqrt.(biases_v_hat) .+ net.params["epsilon"]))
    end
end