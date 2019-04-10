library(yaml)
library(feather)
library(lme4)
library(simr)

cat("\f")
# Read the configuration path
config <- yaml.load_file("./src/config.yml")

#-----------------------------------------------------------------------
# Read the r dataset
eeg_r_dataset_path <- paste("./", config$eeg_r_dataset, sep = "")
df <- read_feather(eeg_r_dataset_path)
df <- na.omit(df)

df
# #-----------------------------------------------------------------------
# # As factor
# df$control_type <- as.factor(df$control_type)
# df$hand_type <- as.factor(df$hand_type)
# df$subject <- as.factor(df$subject)
# 
# #-----------------------------------------------------------------------
# response <- df$theta_alpha
# 
# #-----------------------------------------------------------------------
# # Prior analysis
# engagement <- lmer(response ~ 1 + (1 | df$hand_type),
#                    data = df
# )
# summary(engagement)
# 
# #-----------------------------------------------------------------------
# # Linear mixed models with hand as random effect
# # Set reference level
# df$control_type <- relevel(df$control_type, "no_force")
# df$hand_type[0]
# df$control_type[0]
# 
# control_type <- df$control_type
# # Linear mixed models
# original_model <- lmer(log(response) ~ control_type + (1 | df$subject) + (1 | df$hand_type),
#                        data = df)
# summary(original_model)
# model1 <- glmer(log(response) ~ control_type + (1 | df$subject) + (1 | df$hand_type),
#                 data = df)
# # # #-----------------------------------------------------------------------
# # Power analysis decrease in slope
# fixef(model1)['control_typeerror_augmentation']<- 0.025
# powerSim(model1, nsim=10)
