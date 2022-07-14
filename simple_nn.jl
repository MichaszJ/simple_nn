using LinearAlgebra, Distributions, Random, Zygote

# module simple_nn
# export CreateNetwork, FeedForward

# neural network interfacing
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
end

function CreateNetwork(setup)
    layers = [Layer(layer[1], layer[2], layer[3]) for layer in setup[2:end]]
    sizes = [layer[1] for layer in setup]
    biases = rand.(Normal(), sizes[2:end])
    weights = rand.(Normal(), sizes[2:end], sizes[1:end-1])
    return Network(layers, sizes, biases, weights)
end

function FeedForward(net::Network, a::Vector{Float32})
    for layer in 1:length(net.weights)
        a = net.layers[layer].activation.(net.layers[layer].operation(net, layer, a))
    end
    
    return a
end

# functions
function dense_layer(net::Network, layer::Int, a::Vector{Float32})
    return vec(a' * net.weights[layer]') .+ net.biases[layer]
end

function sigmoid_activation(z)
    return 1 / (1 + exp(-z))
end

# end