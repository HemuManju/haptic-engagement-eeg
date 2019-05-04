library(yaml)
library(feather)
library(car)
library(fitdistrplus)
library(MASS)

# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
eeg_r_dataset_path <- paste('./',config$eeg_r_dataset, sep='')
df <- read_feather(eeg_r_dataset_path)



# Verify the distribution of features 
feature <- df$theta_alpha
qqp(feature, "lnorm")

gamma <- fitdistr(feature, "gamma")
qqp(feature, "gamma", shape = gamma$estimate[[1]], rate = gamma$estimate[[2]])

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
