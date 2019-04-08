library(yaml)
library(feather)
library(ggplot2)
library(ggpubr)

# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
r_dataset_path <- paste('./',config$r_dataset, sep='')
df <- read_feather(r_dataset_path)

comparison_list <- list( c("no_force", "error_augmentation"), 
                        c("no_force", "error_reduction"), 
                        c("error_reduction", "error_augmentation") )
p <- ggboxplot(df, x = 'control_type', y = 'beta_alpha_theta',
          fill = 'control_type', palette = 'jco', bxp.errorbar=TRUE, 
          bxp.errorbar.width = 0.25, width = 0.4)+ 
  stat_compare_means(comparisons = comparison_list, 
                     label.y = c(0.05, 0.125, 0.175)+max(df$beta_alpha_theta))
p 