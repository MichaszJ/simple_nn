using LinearAlgebra, Distributions, Random, Zygote
struct Layer
    size::Int
    operation::Function
    activation::Function
end

mutable struct Network
    layers::Vector{Layer}
    sizes::Vector{Int}
    biases::Vector{Vector{Float32}}
    weights::Vector{Matrix{Float32}}
    params::Dict
end

function CreateNetwork(setup)
    layers = [Layer(layer[1], layer[2], layer[3]) for layer in setup[2:end]]
    sizes = [layer[1] for layer in setup]
    biases = rand.(Normal(), sizes[2:end])
    weights = rand.(Normal(), sizes[2:end], sizes[1:end-1])
    return Network(layers, sizes, biases, weights, Dict())
end

function FeedForward(net::Network, a::Vector{Float32})
    for layer in 1:length(net.weights)
        a = net.layers[layer].activation.(net.layers[layer].operation(net, layer, a))
    end
    
    return a
end

function OptimizerSetup!(net::Network, optimizer; optimizer_params...)
    optimizer_params = get_optimizer_params(net, optimizer; optimizer_params...)
    net.params = optimizer_params
end

function Backward!(net::Network, loss_function, x, y)
    grad = gradient(Params([net.weights, net.biases])) do
        loss_function(x, y)
    end
    
    net.params["optimizer"](net, grad)
end

# functions
# layer types
function dense_layer(net::Network, layer::Int, a::Vector{Float32})
    return vec(a' * net.weights[layer]') .+ net.biases[layer]
end

# activation functions
function none_activation(z)
    return z
end

function sigmoid_activation(z)
    return 1 / (1 + exp(-z))
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
    end
    
    if length(optimizer_params) > 0
        for param in optimizer_params
            params[string(param[1])] = param[2]
        end
    end
    
    return params
end

function GradientDescentOptimizer!(net::Network, grad)
    net.weights = net.weights .- net.params["learning_rate"] * grad[net.weights]
    net.biases = net.biases .- net.params["learning_rate"] * grad[net.biases]
end

function MomentumOptimizer!(net::Network, grad)
    net.params["weights_momentum_vector"] = net.params["gamma"] * net.params["weights_momentum_vector"] + net.params["learning_rate"] * grad[net.weights]
    net.weights = net.weights .- net.params["weights_momentum_vector"]
    
    net.params["biases_momentum_vector"] = net.params["gamma"] * net.params["biases_momentum_vector"] + net.params["learning_rate"] * grad[net.biases]
    net.biases = net.biases .- net.params["biases_momentum_vector"]
end