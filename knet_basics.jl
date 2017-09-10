#
# Me Learning Machine Learning
# Chapter 1: Julia and Knet
#
# In summary, I am (loosely) following the Knet tutorial
# (https://github.com/denizyuret/Knet.jl/blob/master/README.md#Tutorial-1)
# and adding lots of comments, including some of these noob-style "this line
# does x, y and z" comments (because I am a Julia noob :-) ).
#
# Leandro Motta Barros
#

using Knet

# ------------------------------------------------------------------------------
# Reading and pre-processing data
# ------------------------------------------------------------------------------

# This is based on the classic "Boston Housing" dataset, which I downloaded from
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
# (see the file `housing.names` on the same server and directory for more info).
# This database contains 506 lines by 14 columns.

# Read the data into an array. By default, `readdlm` assumes that fields are
# separated by blanks -- quite handy, 'cause that's exactly what we have here.
rawdata = readdlm("housing.data") # [506 x 14]

# Take only the input (independent variables), which is everything but the 14th
# column. We also transpose the matrix, which at first seemed weird for me. It
# turns out that Julia stores arrays in column-major order, so having the data
# transposed will make it more efficient to iterate over the training examples.
x = rawdata[:,1:13]' # [13 x 506]

# Normalization of input variables. The operation itself is pretty standard,
# though the implementation might deserve a comment or two. Here (as in the
# previous line, indeed), there is a lot of Matlab/Octave taste. "Dot-operator"
# to execute element-wise operations. Second parameter to `mean` and `std` (`2`)
# meaning that the mean and standard deviation are to be made along the second
# dimension of the array. In other words, along all elements in a single column.
# Which, in still other words, is the same as saying that we are computing the
# mean and standard deviation among all values of each feature (recall that `x`
# was transposed in the previous line).
x = (x .- mean(x,2)) ./ std(x,2) # [13 x 506]

# Now take the expected output, which is on the 14th (and last) column in our
# data. Again, transpose.
y = rawdata[:,14:14]'

# ------------------------------------------------------------------------------
# Linear regression
# ------------------------------------------------------------------------------

# Initialize the weights. `randn` generates Gaussian-distributed random numbers
# with mean equals to zero and standard deviation of 1.0.

# TODO: Why this explicit array of `Any`? Why that additional zero?

w = Any[ 0.1 * randn(1,13), 0 ]



function train(w, data; lr = .1)
    for (x,y) in data
        dw = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] -= lr * dw[i]
        end
    end
    return w
end


show(w)

#print(x)
