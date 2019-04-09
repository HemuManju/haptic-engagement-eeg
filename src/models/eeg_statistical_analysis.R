library(yaml)
library(feather)
library(lme4)
library(simr)

# Read the configuration path
config <- yaml.load_file("./src/config.yml")

#-----------------------------------------------------------------------
# Read the r dataset
r_dataset_path <- paste("./", config$r_dataset, sep = "")
df <- read_feather(r_dataset_path)
df <- na.omit(df)
#-----------------------------------------------------------------------
# As factor
df$control_type <- as.factor(df$control_type)
df$hand_type <- as.factor(df$hand_type)
df$subject <- as.factor(df$subject)

#-----------------------------------------------------------------------
response <- df$beta_alpha_theta

#-----------------------------------------------------------------------
# Prior analysis
engagement <- lmer(response ~ 1 + (1 | df$hand_type),
  data = df
)
summary(engagement)

#-----------------------------------------------------------------------
# Linear mixed models with hand as random effect
# Set reference level
df$control_type <- relevel(df$control_type, "no_force")
df$hand_type[0]
df$control_type[0]

control_type <- df$control_type
# Linear mixed models
fit.model <- lmer(response ~ control_type + (1 | df$subject) + (1 | df$hand_type), 
                         data = df)
summary(fit.model)
#-----------------------------------------------------------------------
# Power analysis?
power <- powerSim(fit.model, nsim = 10)
power
lastResult()$errors