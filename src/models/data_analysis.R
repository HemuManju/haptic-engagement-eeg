library(yaml)
library(feather)
library(car)
library(fitdistrplus)
library(MASS)

# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
r_dataset_path <- paste('./',config$r_dataset, sep='')
df <- read_feather(r_dataset_path)

# Verify the distribution of features 
feature <- df$beta_alpha_theta

fit.weibull <- fitdist(feature, "weibull")
fit.norm <- fitdist(feature, "norm")
fit.gamma <- fitdist(feature, "gamma")
fit.lnorm <- fitdist(feature, "lnorm")

# # Verify the best fit of data
descdist(feature, discrete = FALSE)
fit.norm$aic
fit.weibull$aic
fit.lnorm$aic
fit.gamma$aic
