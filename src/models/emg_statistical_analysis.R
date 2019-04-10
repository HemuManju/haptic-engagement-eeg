library(yaml)
library(feather)
library(lme4)
library(simr)

# Read the configuration path
config <- yaml.load_file("./src/config.yml")

#-----------------------------------------------------------------------
# Read the r dataset
emg_r_dataset_path <- paste("./", config$emg_r_dataset, sep = "")
df <- read_feather(emg_r_dataset_path)
df <- na.omit(df)
df
#-----------------------------------------------------------------------
# As factor
df$control_type <- as.factor(df$control_type)
df$hand_type <- as.factor(df$hand_type)
df$subject <- as.factor(df$subject)
df
#-----------------------------------------------------------------------
response <- df$slope_zero_crosses

#-----------------------------------------------------------------------
# Prior analysis
prior.fit <- lmer(response ~ 1 + (1 | df$hand_type),
                   data = df
)
summary(prior.fit)

#-----------------------------------------------------------------------
# Linear mixed models with hand as random effect
# Set reference level
df$control_type <- relevel(df$control_type, "no_force")

# Linear mixed models
fit.model <- lmer(response ~ df$control_type + (1 | df$subject),
                  data = df)
summary(fit.model)