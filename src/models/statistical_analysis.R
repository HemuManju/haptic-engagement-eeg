library(yaml)
library(feather)
library(lme4)

# Read the configuration path
path <- paste(getwd(),'/src/config.yml', sep='')
config <- yaml.load_file(path)

# Read the r dataset
r_dataset_path <- paste(getwd(),'/',config$r_dataset, sep='')
df <- read_feather(r_dataset_path)

# Convert the string data into numeric
df$control_type[df$control_type == config$control_type[1]] <- 0
df$control_type[df$control_type == config$control_type[2]] <- -1
df$control_type[df$control_type == config$control_type[3]] <- 1


# Hand type
df$hand_type[df$hand_type == config$hand_type[1]] <- -1
df$hand_type[df$hand_type == config$hand_type[2]] <- 1

# As factor

df$control_type <- as.factor(df$control_type)
df$hand_type <- as.factor(df$hand_type)
# As numeric
df$subject <- as.numeric(df$subject)


# Linear mixed models
engagement.null <- lmer(df$theta_alpha ~ 1 + (1|df$subject),
                            data = df, REML = FALSE)
engagement.hand_type <- lmer(df$theta_alpha ~ df$hand_type + (1|df$subject),
                        data = df, REML = FALSE)
engagement.control_type <- lmer(df$theta_alpha ~ df$control_type + (1|df$subject),
                            data = df, REML = FALSE)
engagement.full_model <- lmer(df$theta_alpha ~ df$hand_type + df$control_type + (1|df$subject),
  data = df, REML = FALSE)


# Null hypothesis testing
anova(engagement.null, engagement.full_model)
anova(engagement.hand_type, engagement.full_model)
anova(engagement.control_type, engagement.full_model)
