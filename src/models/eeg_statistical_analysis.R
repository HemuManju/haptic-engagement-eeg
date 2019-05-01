library(yaml)
library(feather)
library(lme4)
library(simr)

# Read the configuration file -------------------------------------------------------
config <- yaml.load_file("./src/config.yml")


# Read the r dataframe --------------------------------------------------------------
eeg_r_dataset_path <- paste("./", config$eeg_r_dataset, sep = "")
df <- read_feather(eeg_r_dataset_path)
df <- na.omit(df)
config$epoch_length

df


# Convert to factor -----------------------------------------------------------------
df$control_type <- as.factor(df$control_type)
df$hand_type <- as.factor(df$hand_type)
df$subject <- as.factor(df$subject)


# Response --------------------------------------------------------------------------
response <- df$beta_theta


# Prior Analysis --------------------------------------------------------------------
engagement <- lmer(log(response) ~ 1 + (1 | hand_type), data = df)
summary(engagement)


# Linear models 
# Set reference levels --------------------------------------------------------------
df$control_type <- relevel(df$control_type, "no_force")
df$hand_type <- relevel(df$hand_type, "non_dominant")

# Factors
control_type <- df$control_type
hand_type <- df$hand_type
subject <- df$subject


# Linear mixed model 1 --------------------------------------------------------------
model1 <- lmer(log(response) ~ control_type + (1 | subject / hand_type),
               data = df
)
coef(model1)
summary(model1)

# Linear mixed model 2 --------------------------------------------------------------
model2 <- glmer(response ~ control_type + (1 | subject / hand_type),
                data = df, family = Gamma(link = log))
summary(model2)

# Linear mixed model 3 --------------------------------------------------------------
model3 <- glmmPQL(response ~ control_type,
                  random = ~ 1 | subject / hand_type,
                  family = Gamma(link = log), data = df)
summary(model3)


# Power analysis --------------------------------------------------------------------
# fixef(model2)["control_typeerror_augmentation"] <- exp(0.05)
# powerSim(model2, nsim = 10)

