library(yaml)
library(feather)
library(ggplot2)
library(ggpubr)

# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
eeg_r_dataset_path <- paste('./',config$eeg_r_dataset, sep='')
df <- read_feather(eeg_r_dataset_path)

comparison_list <- list( c("no_force", "error_augmentation"), 
                        c("no_force", "error_reduction"), 
                        c("error_reduction", "error_augmentation") )
p <- ggboxplot(df, x = 'control_type', y = 'theta_alpha',
          fill = 'control_type', palette = 'jco', bxp.errorbar=TRUE, 
          bxp.errorbar.width = 0.25, width = 0.4, outline=FALSE)+ 
  # geom_boxplot(outlier.shape = NA) +
  # scale_y_continuous(limits = quantile(df$theta_alpha, c(0.1, 0.9))) +
  stat_compare_means(comparisons = comparison_list, 
                     label.y = c(5, 7.25, 10.175)+max(df$theta_alpha))
p 