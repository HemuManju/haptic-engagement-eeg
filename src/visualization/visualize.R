library(yaml)
library(feather)
library(ggpubr)

# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
r_dataset_path <- paste('./',config$r_dataset, sep='')
df <- read_feather(r_dataset_path)

response <- df$beta_alpha_theta
control <- df$control_type

my_comparisons <- list( c("no_force", "error_augmentation"), 
                        c("no_force", "error_reduction"), 
                        c("error_reduction", "error_augmentation") )
ggboxplot(df, x = 'control_type', y = 'beta_alpha_theta',
          color = 'control_type', palette = "jco")+ 
  stat_compare_means(comparisons = my_comparisons, label.y = c(0.05, 0.125, 0.175)+max(response))