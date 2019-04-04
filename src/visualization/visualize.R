library(yaml)
library(feather)
library(ggplot2)


# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
r_dataset_path <- paste('./',config$r_dataset, sep='')
df <- read_feather(r_dataset_path)

response <- df$beta_alpha_theta
control <- df$control_type

# Set a unique color with fill, colour, and alpha
ggplot(df, aes(x=control, y=response)) + 
  geom_boxplot(color="red", fill="orange", alpha=0.2)

# Set a different color for each group
ggplot(df, aes(x=control, y=response, fill=control)) + 
  geom_boxplot(alpha=0.3, notch = TRUE) +
  theme(legend.position="none")