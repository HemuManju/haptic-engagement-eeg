library(yaml)
library(feather)
library(car)
library(MASS)

# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
r_dataset_path <- paste('./',config$r_dataset, sep='')
df <- read_feather(r_dataset_path)

# Verify the distribution of features 
feature <- df$beta_alpha_theta
qqp(feature, "norm") # normal distribution 
# qqp(feature, "lnorm") # lognormal distribution
# 
# poisson <- fitdistr(feature, "Poisson")
# qqp(feature, "pois", poisson$estimate)
