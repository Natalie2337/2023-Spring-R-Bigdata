### Lab: Deep Learning

## In this version of the Ch10 lab, we  use the `luz` package, which interfaces to the
## `torch` package which in turn links to efficient
## `C++` code in the LibTorch library.

## This version of the lab was produced by Daniel Falbel and Sigrid
## Keydana, both data scientists at Rstudio where these packages were
## produced.

## An advantage over our original `keras` implementation is that this
## version does not require a separate `python` installation.

## Single Layer Network on Hitters Data


###
library(ISLR2)
Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)
###

###
lfit <- lm(Salary ~ ., data = Gitters[-testid, ])
lpred <- predict(lfit, Gitters[testid, ])
with(Gitters[testid, ], mean(abs(lpred - Salary)))
###


###
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
y <- Gitters$Salary
###


###
library(glmnet)
cvfit <- cv.glmnet(x[-testid, ], y[-testid],
    type.measure = "mae")
cpred <- predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred))
###

###
library(torch)
library(luz) # high-level interface for torch
library(torchvision) # for datasets and image transformation
library(torchdatasets) # for datasets we are going to use
library(zeallot)
torch_manual_seed(13)
###

###
modnn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, 50)
    self$activation <- nn_relu()
    self$dropout <- nn_dropout(0.4)
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden() %>%
      self$activation() %>%
      self$dropout() %>%
      self$output()
  }
)
###The object modnn has a single hidden layer with 50 hidden units, 
#and a ReLU activation function. It then has a dropout layer, 
#in which a random 40% of the 50 activations from the previous layer are set to zero 
#during each iteration of the stochastic gradient descent algorithm.

#The pipe operator %>% passes the previous term 
# as the first argument to the next function, and returns the result

###
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
###

#We first make a matrix, and then we standardize each of the variables. 
#We could have obtained the same result using the pipe operator
###
x <- model.matrix(Salary ~ . - 1, data = Gitters) %>% scale()
###

###
modnn <- modnn %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(input_size = ncol(x))
### minimize squared-error

###
fitted <- modnn %>%
  fit(
    data = list(x[-testid, ], matrix(y[-testid], ncol = 1)),
    valid_data = list(x[testid, ], matrix(y[testid], ncol = 1)),
    epochs = 20
  )
#an epoch amounts to the number of SGD steps required to process n observations, n/batch size=epoch
###
plot(fitted)
###



###
npred <- predict(fitted, x[testid, ])
mean(abs(y[testid] - npred))
###

## Multilayer Network on the MNIST Digit Data


###
train_ds <- mnist_dataset(root = ".", train = TRUE, download = TRUE)
test_ds <- mnist_dataset(root = ".", train = FALSE, download = TRUE)

str(train_ds[1])
str(test_ds[2])

length(train_ds)
length(test_ds)
###


###
transform <- function(x) {
  x %>%
    torch_tensor() %>%
    torch_flatten() %>%
    torch_div(255)
}
train_ds <- mnist_dataset(
  root = ".",
  train = TRUE,
  download = TRUE,
  transform = transform
)
test_ds <- mnist_dataset(
  root = ".",
  train = FALSE,
  download = TRUE,
  transform = transform
)
### There are 60,000 images in the training data and 10,000 in the test data. 
#The images are 28??28, and stored as a three-dimensional array, so we need to reshape them into a matrix


###
modelnn <- nn_module(
  initialize = function() {
    self$linear1 <- nn_linear(in_features = 28*28, out_features = 256)
    self$linear2 <- nn_linear(in_features = 256, out_features = 128)
    self$linear3 <- nn_linear(in_features = 128, out_features = 10)

    self$drop1 <- nn_dropout(p = 0.4)
    self$drop2 <- nn_dropout(p = 0.3)

    self$activation <- nn_relu()
  },
  forward = function(x) {
    x %>%

      self$linear1() %>%
      self$activation() %>%
      self$drop1() %>%

      self$linear2() %>%
      self$activation() %>%
      self$drop2() %>%

      self$linear3()
  }
)
###


###
print(modelnn())
###


###
modelnn <- modelnn %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_accuracy())
  )
###

###
system.time(
   fitted <- modelnn %>%
      fit(
        data = train_ds,
        epochs = 5,
        valid_data = 0.2,
        dataloader_options = list(batch_size = 256),
        verbose = FALSE
      )
 )
plot(fitted)
###

###
accuracy <- function(pred, truth) {
   mean(pred == truth) }

# gets the true classes from all observations in test_ds.
truth <- sapply(seq_along(test_ds), function(x) test_ds[x][[2]])

fitted %>%
  predict(test_ds) %>%
  torch_argmax(dim = 2) %>%  # the predicted class is the one with higher 'logit'.
  as_array() %>% # we convert to an R object
  accuracy(truth)
###

###
modellr <- nn_module(
  initialize = function() {
    self$linear <- nn_linear(784, 10)
  },
  forward = function(x) {
    self$linear(x)
  }
)
print(modellr())
###

###
fit_modellr <- modellr %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_accuracy())
  ) %>%
  fit(
    data = train_ds,
    epochs = 5,
    valid_data = 0.2,
    dataloader_options = list(batch_size = 128)
  )

fit_modellr %>%
  predict(test_ds) %>%
  torch_argmax(dim = 2) %>%  # the predicted class is the one with higher 'logit'.
  as_array() %>% # we convert to an R object
  accuracy(truth)


# alternatively one can use the `evaluate` function to get the results
# on the test_ds
evaluate(fit_modellr, test_ds)
###

### Convolutional Neural Networks

###
transform <- function(x) {
  transform_to_tensor(x)
}

train_ds <- cifar100_dataset(
  root = "./",
  train = TRUE,
  download = TRUE,
  transform = transform
)

test_ds <- cifar100_dataset(
  root = "./",
  train = FALSE,
  transform = transform
)

str(train_ds[1])
length(train_ds)
###


###
par(mar = c(0, 0, 0, 0), mfrow = c(5, 5))
index <- sample(seq(50000), 25)
for (i in index) plot(as.raster(as.array(train_ds[i][[1]]$permute(c(2,3,1)))))
### The as.raster() function converts the feature map so that it can be plotted as.raster() as a color image.

###
conv_block <- nn_module(
  initialize = function(in_channels, out_channels) {
    self$conv <- nn_conv2d(
      in_channels = in_channels,
      out_channels = out_channels,
      kernel_size = c(3,3),
      padding = "same"
    )
    self$relu <- nn_relu()
    self$pool <- nn_max_pool2d(kernel_size = c(2,2))
  },
  forward = function(x) {
    x %>%
      self$conv() %>%
      self$relu() %>%
      self$pool()
  }
)

#We use a 3 ?? 3 convolution filter for each channel in all the layers. 
#Each convolution is followed by a maxpooling layer over 2 ?? 2 blocks.

model <- nn_module(
  initialize = function() {
    self$conv <- nn_sequential(
      conv_block(3, 32),
      conv_block(32, 64),
      conv_block(64, 128),
      conv_block(128, 256)
    )
    self$output <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(2*2*256, 512),
      nn_relu(),
      nn_linear(512, 100)
    )
  },
  forward = function(x) {
    x %>%
      self$conv() %>%
      torch_flatten(start_dim = 2) %>%
      self$output()
  }
)
model()
###  After the last of these we have a layer with 256 channels of dimension 2 ?? 2. 
#These are then flattened to a dense layer of size 1,024:
# in other words, each of the 2 ?? 2 matrices is turned into a 4-vector, and put side-by-side in one layer. 
# This is followed by a dropout regularization layer, then another dense layer of size 512

###
fitted <- model %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_accuracy())
  ) %>%
  set_opt_hparams(lr = 0.001) %>%
  fit(
    train_ds,
    epochs = 10, #30,
    valid_data = 0.2,
    dataloader_options = list(batch_size = 128)
  )

print(fitted)

evaluate(fitted, test_ds)
###

