using LinearAlgebra, Distributions, Random, Zygote
struct Layer
    size::Int
    operation::Function
    activation::Function
end

mutable struct Network
    layers::Vector{Layer}
    sizes::Vector{Int}
    biases::Vector
    weights::Vector
    params::Dict
end

function CreateNetwork(setup; datatype=Float32, init_distribution=Normal())
    layers = [Layer(layer[1][2], layer[2], layer[3]) for layer in setup]
    sizes = vcat([setup[1][1][1]], [layer[1][2] for layer in setup])
    biases = [convert.(datatype, layer) for layer in rand.(init_distribution, sizes[2:end])]
    weights = [convert.(datatype, layer) for layer in rand.(init_distribution, sizes[2:end], sizes[1:end-1])]
    return Network(layers, sizes, biases, weights, Dict(["datatype => datatype"]))
end

function Forward(net::Network, a::Vector{Float32})
    for layer in 1:length(net.weights)
        a = net.layers[layer].activation(net.layers[layer].operation(net, layer, a))
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
function dense_layer(net::Network, layer::Int, a)
    return vec(a' * net.weights[layer]') .+ net.biases[layer]
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