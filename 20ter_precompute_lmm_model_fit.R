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





################
#### Cxy ####
################

root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Analyses/precompute/allsujet/PSD_Coh"
outputdir = paste(root, "stats", sep = "/")

channels <- c('Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
            'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2')
sujet_groups <- c('allsujet', 'rep', 'norep')

# Load the Excel data
df_raw <- read_excel(paste(root, "Cxy_allsujet.xlsx", sep  = "/"))

#### allsujet, rep, norep

chan_sel = "Fp1"
sujet_group = "norep"

for (sujet_group in sujet_groups) {

  for (chan_sel in channels) {
    
    print(chan_sel)
    print(sujet_group)
    
    if (sujet_group == "allsujet") {
      df_sujet_sel = df_raw
    }
    if (sujet_group == "rep") {
      df_sujet_sel = subset(df_raw, REP == TRUE)
    }
    if (sujet_group == "norep") {
      df_sujet_sel = subset(df_raw, REP == FALSE)
    }
  
    df_onechan <- subset(df_sujet_sel, chan == chan_sel)
    
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
        title    = paste(sujet_group, chan_sel, "Cxy", sep = "_")
      ) +
      theme(
        plot.title    = element_text(hjust = 0.5),
      )
    
    p
    
    file_boxplot_subjectwise = paste(sujet_group, "boxplot", chan_sel, "Cxy_subjectwise.png", sep = "_")
    # then explicitly:
    ggsave(paste(outputdir, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
    
    # model
    complex_form <- Cxy ~ cond * odor + (cond + odor | sujet)
    simple_form  <- Cxy ~ cond * odor + (1 | sujet)
    
    # 1) Attempt the complex fit
    #model <- try(lmer(complex_form, data = df), silent = TRUE)
    
    # 2) If it failed, or if it converged to a singular fit, refit simple
    #if (inherits(model, "try-error") || isSingular(model, tol = 1e-4)) {
    #  message("Complex model failed → refitting simple model.")
    #  model <- lmer(simple_form, data = df)
    #}
    
    #model <- lmer(simple_form, data = df)
    model <- lmer(complex_form, data = df)
    
    filename_hist = paste(sujet_group, "histogram", chan_sel, "Cxy.png", sep = "_")
    
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
      main     = paste(sujet_group, chan_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
      adj      = 0.5,       # 0.5 = center
      cex.main = 1.5,       # main title size
      font.main= 2,         # bold
      cex.sub  = 1.0        # subtitle size
    )
    
    dev.off()
    
    # View summary
    summary(model)
    
    filename_qqplot = paste(sujet_group, "qqplot", chan_sel, "Cxy.png", sep = "_")
    
    png(
      filename = paste(outputdir, filename_qqplot, sep = "/"),
      width    = 800,    # width in pixels
      height   = 600,    # height in pixels
      res      = 100     # resolution (pixels per inch)
    )
    
    qqnorm(resid(model))
    qqline(resid(model))  # points fall nicely onto the line - good!
    
    title(
      sub     = paste(sujet_group, chan_sel, "qqplot"),
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
    filesxlsx_chan = paste(sujet_group, "lmm", chan_sel, "res.xlsx", sep = "_")
    writexl::write_xlsx(model_df, paste(outputdir, filesxlsx_chan, sep = "/"))
    
  }

}

#### repnorep

chan_sel = "Fp1"
sujet_group = "repnorep"

for (chan_sel in channels) {
  
  print(chan_sel)
  print(sujet_group)
  
  df_sujet_sel = df_raw

  
  df_onechan <- subset(df_sujet_sel, chan == chan_sel)
  
  df <- df_onechan
  
  # See unique value
  #lapply(df_chunk, unique)
  
  # Convert categorical variables to factors
  df$sujet <- as.factor(df$sujet)
  df$cond <- as.factor(df$cond)
  df$odor <- as.factor(df$odor)
  df$REP <- as.factor(df$REP)
  
  df$cond <- relevel(df$cond, ref = "FR_CV_1")  
  df$odor <- relevel(df$odor, ref = "o")  
  df$REP <- relevel(df$REP, ref = "FALSE")  
  
  p <- ggplot(df, aes(x = sujet, y = Cxy, color = sujet, fill = sujet)) +
    geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                 position = position_dodge(.9)) +
    stat_summary(fun = median, geom = "point", size = 2,
                 position = position_dodge(.9), color = "white")+
    labs(
      title    = paste(sujet_group, chan_sel, "Cxy", sep = "_")
    ) +
    theme(
      plot.title    = element_text(hjust = 0.5),
    )
  
  p
  
  file_boxplot_subjectwise = paste(sujet_group, "boxplot", chan_sel, "Cxy_subjectwise.png", sep = "_")
  # then explicitly:
  ggsave(paste(outputdir, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
  
  # model
  complex_form <- Cxy ~ cond * odor * REP + (cond + odor | sujet)
  simple_form  <- Cxy ~ cond * odor + (1 | sujet)
  
  # 1) Attempt the complex fit
  #model <- try(lmer(complex_form, data = df), silent = TRUE)
  
  # 2) If it failed, or if it converged to a singular fit, refit simple
  #if (inherits(model, "try-error") || isSingular(model, tol = 1e-4)) {
  #  message("Complex model failed → refitting simple model.")
  #  model <- lmer(simple_form, data = df)
  #}
  
  #model <- lmer(simple_form, data = df)
  model <- lmer(complex_form, data = df)
  
  filename_hist = paste(sujet_group, "histogram", chan_sel, "Cxy.png", sep = "_")
  
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
    main     = paste(sujet_group, chan_sel, "Cxy", "kurtosis:", kurt_chan, "skewness", skew_chan),
    adj      = 0.5,       # 0.5 = center
    cex.main = 1.5,       # main title size
    font.main= 2,         # bold
    cex.sub  = 1.0        # subtitle size
  )
  
  dev.off()
  
  # View summary
  summary(model)
  
  filename_qqplot = paste(sujet_group, "qqplot", chan_sel, "Cxy.png", sep = "_")
  
  png(
    filename = paste(outputdir, filename_qqplot, sep = "/"),
    width    = 800,    # width in pixels
    height   = 600,    # height in pixels
    res      = 100     # resolution (pixels per inch)
  )
  
  qqnorm(resid(model))
  qqline(resid(model))  # points fall nicely onto the line - good!
  
  title(
    sub     = paste(sujet_group, chan_sel, "qqplot"),
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
  filesxlsx_chan = paste(sujet_group, "lmm", chan_sel, "res.xlsx", sep = "_")
  writexl::write_xlsx(model_df, paste(outputdir, filesxlsx_chan, sep = "/"))
  
}







############
#### TF ####
############

root = "Z:/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Analyses/precompute/allsujet/TF"
root = "/home/jules/smb4k/CRNLDATA/crnldata/cmo/Projets/Olfadys/NBuonviso2022_jules_olfadys/EEG_Paris_J/Analyses/precompute/allsujet/TF"
outputdir = paste(root, "lmm", sep = "/")

channels <- c('Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 
              'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2')
sujet_groups <- c('allsujet', 'rep', 'norep')
band_list <- c('theta', 'alpha', 'beta', 'gamma')
phase_list <- c('inspi', 'expi')
region_list <- c('frontal', 'parietal', 'temporal', 'occipital', 'central')

# Load the Excel data
df_raw <- read_excel(paste(root, "df_allsujet_TF.xlsx", sep  = "/"))
df_raw_region <- read_excel(paste(root, "df_allsujet_TF_region.xlsx", sep  = "/"))

#### allsujet rep norep

# See unique value
#lapply(df_raw, unique)

chan_sel = "C3"
band_sel = 'theta'
sujet_group = "allsujet"
phase_sel = "inspi"

for (sujet_group in sujet_groups) {
  
  for (chan_sel in channels) {
    
    for (band_sel in band_list) {
      
      for (phase_sel in phase_list) {
        
        print(chan_sel)
        print(sujet_group)
        print(band_sel)
        print(phase_sel)
        
        if (sujet_group == "allsujet") {
          df_sujet_sel = df_raw
        }
        if (sujet_group == "rep") {
          df_sujet_sel = subset(df_raw, rep == TRUE)
        }
        if (sujet_group == "norep") {
          df_sujet_sel = subset(df_raw, rep == FALSE)
        }
        
        df_onechan <- subset(df_sujet_sel, chan == chan_sel & band == band_sel & phase == phase_sel)
        
        df <- df_onechan[c("sujet", "cond", "odor", "phase", "Pxx")]
        
        # See unique value
        #lapply(df_raw, unique)
        
        # Convert categorical variables to factors
        df$sujet <- as.factor(df$sujet)
        df$cond <- as.factor(df$cond)
        df$odor <- as.factor(df$odor)
        
        df$cond <- relevel(df$cond, ref = "FR_CV_1")  
        df$odor <- relevel(df$odor, ref = "o")  
        
        p <- ggplot(df, aes(x = sujet, y = Pxx, color = sujet, fill = sujet)) +
          geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                       position = position_dodge(.9)) +
          stat_summary(fun = median, geom = "point", size = 2,
                       position = position_dodge(.9), color = "white")+
          labs(
            title    = paste(sujet_group, band_sel, chan_sel, phase_sel, "Pxx", sep = "_")
          ) +
          theme(
            plot.title    = element_text(hjust = 0.5),
          )
        
        p
        
        file_boxplot_subjectwise = paste(sujet_group, "boxplot", band_sel, chan_sel, phase_sel, "Pxx_subjectwise.png", sep = "_")
        # then explicitly:
        ggsave(paste(outputdir, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
        
        # model
        complex_form <- Pxx ~ cond * odor + (cond + odor | sujet)
        simple_form  <- Pxx ~ cond * odor + (1 | sujet)
        
        # 1) Attempt the complex fit
        #model <- try(lmer(complex_form, data = df), silent = TRUE)
        
        # 2) If it failed, or if it converged to a singular fit, refit simple
        #if (inherits(model, "try-error") || isSingular(model, tol = 1e-4)) {
        #  message("Complex model failed → refitting simple model.")
        #  model <- lmer(simple_form, data = df)
        #}
        
        #model <- lmer(simple_form, data = df)
        model <- lmer(complex_form, data = df)
        
        filename_hist = paste(sujet_group, band_sel, "histogram", chan_sel, phase_sel, "Pxx.png", sep = "_")
        
        skew_chan = round(skewness(df$Pxx), 2)
        kurt_chan = round(kurtosis(df$Pxx), 2)
        
        png(
          filename = paste(outputdir, filename_hist, sep = "/"),
          width    = 800,    # width in pixels
          height   = 600,    # height in pixels
          res      = 100     # resolution (pixels per inch)
        )
        
        hist(
          df$Pxx,
          breaks = 30,
          main   = "",          # leave main blank for now
          xlab   = "Pxx values",
          ylab   = "Frequency",
          col    = "lightblue",
          border = "white"
        )
        
        title(
          main     = paste(sujet_group, band_sel, chan_sel, phase_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
          adj      = 0.5,       # 0.5 = center
          cex.main = 1.5,       # main title size
          font.main= 2,         # bold
          cex.sub  = 1.0        # subtitle size
        )
        
        dev.off()
        
        # View summary
        summary(model)
        
        filename_qqplot = paste(sujet_group, "qqplot", chan_sel, band_sel, phase_sel, "Pxx.png", sep = "_")
        
        png(
          filename = paste(outputdir, filename_qqplot, sep = "/"),
          width    = 800,    # width in pixels
          height   = 600,    # height in pixels
          res      = 100     # resolution (pixels per inch)
        )
        
        qqnorm(resid(model))
        qqline(resid(model))  # points fall nicely onto the line - good!
        
        title(
          sub     = paste(sujet_group, chan_sel, band_sel, phase_sel, "qqplot"),
          adj      = 0.5,       # 0.5 = center
          cex.main = 1.5,       # main title size
          font.main= 2,         # bold
          cex.sub  = 1.0        # subtitle size
        )
        
        dev.off()
        
        ggplot(df, aes(x = interaction(cond, odor), y = Pxx)) +
          geom_boxplot() +
          labs(x = "Condition × Odor", y = "Pxx") +
          theme_minimal()
        
        # Print table for a mixed model
        tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
        
        # Tidy the model
        model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
        
        # Export to xlsx
        filesxlsx_chan = paste(sujet_group, "lmm", chan_sel, band_sel, phase_sel, "res.xlsx", sep = "_")
        writexl::write_xlsx(model_df, paste(outputdir, filesxlsx_chan, sep = "/"))
        
      }
      
    }
    
  }
  
}

#### repnorep

# See unique value
#lapply(df_raw, unique)

chan_sel = "C3"
band_sel = 'theta'
phase_sel = "inspi"
sujet_group = 'repnorep'


for (chan_sel in channels) {
  
  for (band_sel in band_list) {
    
    for (phase_sel in phase_list) {
      
      print(chan_sel)
      print(sujet_group)
      print(band_sel)
      print(phase_sel)
      
      df_sujet_sel = df_raw
      
      df_onechan <- subset(df_sujet_sel, chan == chan_sel & band == band_sel & phase == phase_sel)
      
      df <- df_onechan[c("sujet", "cond", "odor", "phase", "Pxx", "rep")]
      
      # See unique value
      #lapply(df_raw, unique)
      
      # Convert categorical variables to factors
      df$sujet <- as.factor(df$sujet)
      df$cond <- as.factor(df$cond)
      df$odor <- as.factor(df$odor)
      df$rep <- as.factor(df$rep)
      
      df$cond <- relevel(df$cond, ref = "FR_CV_1")  
      df$odor <- relevel(df$odor, ref = "o")  
      df$rep <- relevel(df$rep, ref = "FALSE")  
      
      p <- ggplot(df, aes(x = sujet, y = Pxx, color = sujet, fill = sujet)) +
        geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                     position = position_dodge(.9)) +
        stat_summary(fun = median, geom = "point", size = 2,
                     position = position_dodge(.9), color = "white")+
        labs(
          title    = paste(sujet_group, band_sel, chan_sel, phase_sel, "Pxx", sep = "_")
        ) +
        theme(
          plot.title    = element_text(hjust = 0.5),
        )
      
      p
      
      file_boxplot_subjectwise = paste(sujet_group, "boxplot", band_sel, chan_sel, phase_sel, "Pxx_subjectwise.png", sep = "_")
      # then explicitly:
      ggsave(paste(outputdir, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
      
      # model
      complex_form <- Pxx ~ cond * odor * rep + (cond + odor | sujet)
      simple_form  <- Pxx ~ cond * odor * rep + (1 | sujet)
      
      # 1) Attempt the complex fit
      #model <- try(lmer(complex_form, data = df), silent = TRUE)
      
      # 2) If it failed, or if it converged to a singular fit, refit simple
      #if (inherits(model, "try-error") || isSingular(model, tol = 1e-4)) {
      #  message("Complex model failed → refitting simple model.")
      #  model <- lmer(simple_form, data = df)
      #}
      
      model <- lmer(simple_form, data = df)
      #model <- lmer(complex_form, data = df)
      
      filename_hist = paste(sujet_group, band_sel, "histogram", chan_sel, phase_sel, "Pxx.png", sep = "_")
      
      skew_chan = round(skewness(df$Pxx), 2)
      kurt_chan = round(kurtosis(df$Pxx), 2)
      
      png(
        filename = paste(outputdir, filename_hist, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      hist(
        df$Pxx,
        breaks = 30,
        main   = "",          # leave main blank for now
        xlab   = "Pxx values",
        ylab   = "Frequency",
        col    = "lightblue",
        border = "white"
      )
      
      title(
        main     = paste(sujet_group, band_sel, chan_sel, phase_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      # View summary
      summary(model)
      
      filename_qqplot = paste(sujet_group, "qqplot", chan_sel, band_sel, phase_sel, "Pxx.png", sep = "_")
      
      png(
        filename = paste(outputdir, filename_qqplot, sep = "/"),
        width    = 800,    # width in pixels
        height   = 600,    # height in pixels
        res      = 100     # resolution (pixels per inch)
      )
      
      qqnorm(resid(model))
      qqline(resid(model))  # points fall nicely onto the line - good!
      
      title(
        sub     = paste(sujet_group, chan_sel, band_sel, phase_sel, "qqplot"),
        adj      = 0.5,       # 0.5 = center
        cex.main = 1.5,       # main title size
        font.main= 2,         # bold
        cex.sub  = 1.0        # subtitle size
      )
      
      dev.off()
      
      ggplot(df, aes(x = interaction(cond, odor), y = Pxx)) +
        geom_boxplot() +
        labs(x = "Condition × Odor", y = "Pxx") +
        theme_minimal()
      
      # Print table for a mixed model
      tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
      
      # Tidy the model
      model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
      
      # Export to xlsx
      filesxlsx_chan = paste(sujet_group, "lmm", chan_sel, band_sel, phase_sel, "res.xlsx", sep = "_")
      writexl::write_xlsx(model_df, paste(outputdir, filesxlsx_chan, sep = "/"))
      
    }
    
  }
  
}





#### REGION

# See unique value
#lapply(df_raw, unique)

region_sel = "frontal"
band_sel = 'theta'
sujet_group = "allsujet"
phase_sel = "inspi"

for (sujet_group in sujet_groups) {
  
  for (region_sel in region_list) {
    
    for (band_sel in band_list) {
      
      for (phase_sel in phase_list) {
        
        print(region_sel)
        print(sujet_group)
        print(band_sel)
        print(phase_sel)
        
        if (sujet_group == "allsujet") {
          df_sujet_sel = df_raw_region
        }
        if (sujet_group == "rep") {
          df_sujet_sel = subset(df_raw_region, rep == TRUE)
        }
        if (sujet_group == "norep") {
          df_sujet_sel = subset(df_raw_region, rep == FALSE)
        }
        
        df_onechan <- subset(df_sujet_sel, region == region_sel & band == band_sel & phase == phase_sel)
        
        df <- df_onechan[c("sujet", "cond", "odor", "phase", "Pxx")]
        
        # See unique value
        #lapply(df_raw_region, unique)
        
        # Convert categorical variables to factors
        df$sujet <- as.factor(df$sujet)
        df$cond <- as.factor(df$cond)
        df$odor <- as.factor(df$odor)
        
        df$cond <- relevel(df$cond, ref = "FR_CV_1")  
        df$odor <- relevel(df$odor, ref = "o")  
        
        p <- ggplot(df, aes(x = sujet, y = Pxx, color = sujet, fill = sujet)) +
          geom_boxplot(width = .2, alpha = .5, outlier.alpha = 0,
                       position = position_dodge(.9)) +
          stat_summary(fun = median, geom = "point", size = 2,
                       position = position_dodge(.9), color = "white")+
          labs(
            title    = paste(sujet_group, band_sel, region_sel, phase_sel, "Pxx", sep = "_")
          ) +
          theme(
            plot.title    = element_text(hjust = 0.5),
          )
        
        p
        
        file_boxplot_subjectwise = paste(sujet_group, "boxplot", band_sel, region_sel, phase_sel, "Pxx_subjectwise.png", sep = "_")
        # then explicitly:
        ggsave(paste(outputdir, file_boxplot_subjectwise, sep = "/"), plot = p, width = 8, height = 5)
        
        # model
        complex_form <- Pxx ~ cond * odor + (cond + odor | sujet)
        simple_form  <- Pxx ~ cond * odor + (1 | sujet)
        
        # 1) Attempt the complex fit
        #model <- try(lmer(complex_form, data = df), silent = TRUE)
        
        # 2) If it failed, or if it converged to a singular fit, refit simple
        #if (inherits(model, "try-error") || isSingular(model, tol = 1e-4)) {
        #  message("Complex model failed → refitting simple model.")
        #  model <- lmer(simple_form, data = df)
        #}
        
        model <- lmer(simple_form, data = df)
        #model <- lmer(complex_form, data = df)
        
        filename_hist = paste(sujet_group, band_sel, "histogram", region_sel, phase_sel, "Pxx.png", sep = "_")
        
        skew_chan = round(skewness(df$Pxx), 2)
        kurt_chan = round(kurtosis(df$Pxx), 2)
        
        png(
          filename = paste(outputdir, filename_hist, sep = "/"),
          width    = 800,    # width in pixels
          height   = 600,    # height in pixels
          res      = 100     # resolution (pixels per inch)
        )
        
        hist(
          df$Pxx,
          breaks = 30,
          main   = "",          # leave main blank for now
          xlab   = "Pxx values",
          ylab   = "Frequency",
          col    = "lightblue",
          border = "white"
        )
        
        title(
          main     = paste(sujet_group, band_sel, region_sel, phase_sel, "Pxx", "kurtosis:", kurt_chan, "skewness", skew_chan),
          adj      = 0.5,       # 0.5 = center
          cex.main = 1.5,       # main title size
          font.main= 2,         # bold
          cex.sub  = 1.0        # subtitle size
        )
        
        dev.off()
        
        # View summary
        summary(model)
        
        filename_qqplot = paste(sujet_group, "qqplot", region_sel, band_sel, phase_sel, "Pxx.png", sep = "_")
        
        png(
          filename = paste(outputdir, filename_qqplot, sep = "/"),
          width    = 800,    # width in pixels
          height   = 600,    # height in pixels
          res      = 100     # resolution (pixels per inch)
        )
        
        qqnorm(resid(model))
        qqline(resid(model))  # points fall nicely onto the line - good!
        
        title(
          sub     = paste(sujet_group, region_sel, band_sel, phase_sel, "qqplot"),
          adj      = 0.5,       # 0.5 = center
          cex.main = 1.5,       # main title size
          font.main= 2,         # bold
          cex.sub  = 1.0        # subtitle size
        )
        
        dev.off()
        
        ggplot(df, aes(x = interaction(cond, odor), y = Pxx)) +
          geom_boxplot() +
          labs(x = "Condition × Odor", y = "Pxx") +
          theme_minimal()
        
        # Print table for a mixed model
        tab_model(model, show.re.var = TRUE, show.icc = TRUE, show.r2 = TRUE, show.se = TRUE)
        
        # Tidy the model
        model_df <- broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE)
        
        # Export to xlsx
        filesxlsx_chan = paste(sujet_group, "lmm", region_sel, band_sel, phase_sel, "res.xlsx", sep = "_")
        writexl::write_xlsx(model_df, paste(outputdir, filesxlsx_chan, sep = "/"))
        
      }
      
    }
    
  }
  
}