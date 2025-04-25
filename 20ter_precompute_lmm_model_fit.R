# Required libraries
library(readxl)
library(lme4)
library(lmerTest)  # For p-values
library(emmeans)
library(ggplot2)
library(e1071)  # For skewness and kurtosis
library(pbkrtest)
library(sjPlot)
library(broom.mixed)
library(writexl)


root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Analyses/precompute/allsujet/PSD_Coh"
outputdir = paste(root, "stats", sep = "/")

channels <- c('Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
            'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2')

# Load the Excel data
df_raw <- read_excel(paste(root, "Cxy_allsujet.xlsx", sep  = "/"))

chan_sel = "Fp1"

for (chan_sel in channels) {
  
  print(chan_sel)

  df_onechan <- subset(df_raw, chan == chan_sel)
  
  df <- df_onechan[c("sujet", "cond", "odor", "Cxy")]
  
  # See unique value
  #lapply(df_chunk, unique)
  
  # Convert categorical variables to factors
  df$sujet <- as.factor(df$sujet)
  df$cond <- as.factor(df$cond)
  df$odor <- as.factor(df$odor)
  
  df$cond <- relevel(df$cond, ref = "FR_CV_1")  
  df$odor <- relevel(df$odor, ref = "o")  
  
  p <- ggplot(df, aes(x = sujet, y = Cxy, color = sujet, fill = sujet)) +
    geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                 position = position_dodge(.9)) +
    stat_summary(fun = median, geom = "point", size = 2,
                 position = position_dodge(.9), color = "white")+
    labs(
      title    = paste(chan_sel, "Cxy", sep = "_")
    ) +
    theme(
      plot.title    = element_text(hjust = 0.5),
    )
  
  p
  
  file_boxplot_subjectwise = paste("boxplot", chan_sel, "Cxy_subjectwise.png", sep = "_")
  # then explicitly:
  ggsave(paste(outputdir, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
  
  # model
  complex_form <- Cxy ~ cond * odor + (cond + odor | sujet)
  simple_form  <- Cxy ~ cond * odor + (1 | sujet)
  
  # 1) Attempt the complex fit
  model <- try(lmer(complex_form, data = df), silent = TRUE)
  
  # 2) If it failed, or if it converged to a singular fit, refit simple
  if (inherits(model, "try-error") || isSingular(model, tol = 1e-4)) {
    message("Complex model failed → refitting simple model.")
    model <- lmer(simple_form, data = df)
  }

  filename_hist = paste("histogram", chan_sel, "Cxy.png", sep = "_")
  
  skew_chan = round(skewness(df$Cxy), 2)
  kurt_chan = round(kurtosis(df$Cxy), 2)
  
  png(
    filename = paste(outputdir, filename_hist, sep = "/"),
    width    = 800,    # width in pixels
    height   = 600,    # height in pixels
    res      = 100     # resolution (pixels per inch)
  )
  
  hist(
    df$Cxy,
    breaks = 30,
    main   = "",          # leave main blank for now
    xlab   = "Cxy values",
    ylab   = "Frequency",
    col    = "lightblue",
    border = "white"
  )
  
  title(
    main     = paste(chan_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
    adj      = 0.5,       # 0.5 = center
    cex.main = 1.5,       # main title size
    font.main= 2,         # bold
    cex.sub  = 1.0        # subtitle size
  )
  
  dev.off()
  
  # View summary
  summary(model)
  
  filename_qqplot = paste("qqplot", chan_sel, "Cxy.png", sep = "_")
  
  png(
    filename = paste(outputdir, filename_qqplot, sep = "/"),
    width    = 800,    # width in pixels
    height   = 600,    # height in pixels
    res      = 100     # resolution (pixels per inch)
  )
  
  qqnorm(resid(model))
  qqline(resid(model))  # points fall nicely onto the line - good!
  
  title(
    sub     = paste(chan_sel, "qqplot"),
    adj      = 0.5,       # 0.5 = center
    cex.main = 1.5,       # main title size
    font.main= 2,         # bold
    cex.sub  = 1.0        # subtitle size
  )
  
  dev.off()
  
  ggplot(df, aes(x = interaction(cond, odor), y = Cxy)) +
    geom_boxplot() +
    labs(x = "Condition × Odor", y = "Cxy") +
    theme_minimal()
  
  # Print table for a mixed model
  tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
  
  # Tidy the model
  model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
  
  # Export to xlsx
  filesxlsx_chan = paste("lmm", chan_sel, "res.xlsx", sep = "_")
  writexl::write_xlsx(model_df, paste(outputdir, filesxlsx_chan, sep = "/"))
  
}



