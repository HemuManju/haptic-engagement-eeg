library(yaml)
library(feather)
library(ggplot2)
library(ggpubr)
library(dplyr)

# Read the configuration path
config <- yaml.load_file('./src/config.yml')

# Read the r dataset
eeg_r_dataset_path <- paste('./',config$eeg_r_dataset, sep='')
df <- read_feather(eeg_r_dataset_path)
df <- filter(df, df$subject==config$subjects[4])
df <- filter(df, df$hand_type==config$hand_type[2])

df[, 1:6] <- log(df[1:6], 2)


comparison_list <- list( c("no_force", "error_augmentation"), 
                        c("no_force", "error_reduction"), 
                        c("error_reduction", "error_augmentation") )
p <- ggboxplot(df, x = 'control_type', y = 'beta_alpha',
          fill = 'control_type', palette = 'jco', bxp.errorbar=TRUE, 
          bxp.errorbar.width = 0.25, width = 0.4, outline=FALSE)+ 
  # geom_boxplot(outlier.shape = NA) +
  # scale_y_continuous(limits = quantile(df$theta_alpha, c(0.1, 0.9))) +
  stat_compare_means(comparisons = comparison_list, 
                     label.y = c(0.25, 0.5, 0.75)+max(df$beta_alpha))
p 