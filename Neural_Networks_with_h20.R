# This file uses the h20.ai machine learning software for R to train a neural network to predict 
# Credit card default in the public dataset taken from https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients


# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (! ("jsonlite" %in% rownames(installed.packages()))) { install.packages("jsonlite") }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-turing/8/R")))
library(h2o)
localH2O = h2o.init(nthreads=-1)

# Initialize
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, 
                    Xmx = '2g')
df = read.csv("~/R/CCarddata.csv", header = TRUE, sep = ",")

# Remove ID columns
df <- df[,-1]
df <- df[,-25]

maxs <- apply(df, 2, max) 
mins <- apply(df, 2, min)

df <- as.data.frame(scale(df, center = mins, scale = maxs - mins))

#https://www.r-bloggers.com/things-to-try-after-user-part-1-deep-learning-with-h2o/


card.hex <- as.h2o(df) # Turn into .hex environment
test.hex <- as.h2o(df[1:23])

df <- df[,-c(24,25)]

# Initialize model with hidden layer 60,60

model = h2o.deeplearning(x = 1:23,
                         y = 24,
                         training_frame = card.hex,
                         activation = "RectifierWithDropout",
                         hidden = c(60,60),
                         epochs = 12000)
x = setdiff(colnames(card.hex),
            c("default.payment.next.month"))

model
# Predict on test set
h2o_yhat_test <- h2o.predict(model, test.hex)
h2o_yhat_test
df_yhat_test <- as.data.frame(h2o_yhat_test)

#----------------OUTPUT--------------------

#H2ORegressionMetrics: deeplearning
#** Reported on training data. **
#  ** Metrics reported on temporary training frame with 9946 samples **
  
#MSE:  0.1322842
#RMSE:  0.363709
#MAE:  0.2871171
#RMSLE:  0.2581368
#Mean Residual Deviance :  0.1322842



