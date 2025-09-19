# load packages used in this tutorial
#install.packages("haven") 
#install.packages("tidyverse")
#install.packages("ggeffects")
#install.packages("merTools")
#install.packages("labelled")
#install.packages("sjPlot")
#install.packages("Metrics")
#install.packages("marginaleffects")
#install.packages("sandwich")
#install.packages("clubSandwich")
#install.packages("lmtest")

library(haven)
library(tidyverse)
library(ggeffects)
library(lme4)
library(merTools) 
library(labelled)
library(sjPlot)
library(Metrics)
library(dplyr) 
library(marginaleffects) 
library(sandwich)
library(clubSandwich)
library(lmtest)

# ------------------------------------------------------------------------------
# Load data
data = read.csv("LFS_DoctorBackgroundForMarginalEffects.csv")

# Specify whether to export results
export_csv <- 1

# Create new numerical binary outcome variable of interest
outcome_binary_string = "Flag_CurrentDoctor_1.0"
colnames(data)[colnames(data) == outcome_binary_string] ="outcome_binary"

# Specify which variable to use for weighting
weight_var = "IPW_ProvidedWeightPlusSelection_winsorised"

# ------------------------------------------------------------------------------
### Other dataset specific processing
# Turn numerical variables into factor variables
data$lfs_year <- as.factor(data$lfs_year)
data$year_age18_5yrband_agg_doctoronly <- as.factor(data$year_age18_5yrband_agg_doctoronly)
data$SEX <- as.factor(data$SEX)
data$CRY12 <- as.factor(data$CRY12)
data$countryofbirth_agg <- as.factor(data$countryofbirth_agg)
data$countryofbirth_binary <- as.factor(data$countryofbirth_binary)
data$ETHUKEUL <- as.factor(data$ETHUKEUL)
data$ethnicgroup_category <- as.factor(data$ethnicgroup_category)
data$ethnicgroup_binary <- as.factor(data$ethnicgroup_binary)
data$nssec_familybackground_separatedoctor_9cat <- as.factor(data$nssec_familybackground_separatedoctor_9cat)
data$nssec_familybackground_9cat <- as.factor(data$nssec_familybackground_9cat)
data$nssec_familybackground_separatedoctor_3cat <- as.factor(data$nssec_familybackground_separatedoctor_3cat)
data$nssec_familybackground_3cat <- as.factor(data$nssec_familybackground_3cat)
data$SOC_occupation_familyback_separatedoctor <- as.factor(data$SOC_occupation_familyback_separatedoctor)


# Set reference categories
# LFS year - 2018 as largest
data$lfs_year <- relevel(as.factor(data$lfs_year), "2018")
# Age band - 30-40 as largest
data$year_age18_5yrband_agg_doctoronly <- relevel(as.factor(data$year_age18_5yrband_agg_doctoronly), "15: 1980-1985")
# Sex - Female - "1" as first alphabetically, to avoid reinforcing male as default
data$SEX <- relevel(as.factor(data$SEX), "1")
# Country of birth disaggregated - England/UK as largest
data$CRY12 <- relevel(as.factor(data$CRY12), "921")
data$countryofbirth_agg <- relevel(as.factor(data$countryofbirth_agg), "921")
data$countryofbirth_binary <- relevel(as.factor(data$countryofbirth_binary), "921")
# Ethnic group - White as largest
data$ETHUKEUL <- relevel(as.factor(data$ETHUKEUL), "1")
data$ethnicgroup_category <- relevel(as.factor(data$ethnicgroup_category), "1")
data$ethnicgroup_binary <- relevel(as.factor(data$ethnicgroup_binary), "1")
# Family background (without doctor separated) - 1.1 as highest income
data$nssec_familybackground_9cat <- relevel(as.factor(data$nssec_familybackground_9cat), "1.1")
# Family background (with doctor separated) - Doctor
data$nssec_familybackground_separatedoctor_9cat <- relevel(as.factor(data$nssec_familybackground_separatedoctor_9cat), "Doctor")
# Family background, 3-cat (without doctor separated) - 1-2 as highest income
data$nssec_familybackground_3cat <- relevel(as.factor(data$nssec_familybackground_3cat), "1-2")
# Family background, 3-cat (with doctor separated) - Doctor
data$nssec_familybackground_separatedoctor_3cat <- relevel(as.factor(data$nssec_familybackground_separatedoctor_3cat), "Doctor")
# Main earner occupation - Doctor, code = 2211
data$SOC_occupation_familyback_separatedoctor <- relevel(as.factor(data$SOC_occupation_familyback_separatedoctor), "2211")


# ------------------------------------------------------------------------------
# Exposure = NS-SEC 9-cat with doctor separated
# ------------------------------------------------------------------------------
# Estimate Fixed-Effects only Logistic Regression model
# Specify exposure variable
exposure_var = "nssec_familybackground_separatedoctor_9cat"

# Adapted from https://www.statology.org/r-logistic-regression-predict/
# Fit the fixed-effects only logistic regression
model_fixedeffects_9catdocsep <- glm(outcome_binary ~ lfs_year + year_age18_5yrband_agg_doctoronly + SEX + CRY12 + ETHUKEUL + nssec_familybackground_separatedoctor_9cat, 
                                     data=data, 
                                     weights = IPW_ProvidedWeightPlusSelection_winsorised,
                                     family=binomial # logistic regression
)

#view model summary
summary(model_fixedeffects_9catdocsep)

# Get coefficients with HC3 robust standard errors (following https://stats.stackexchange.com/questions/117052/replicating-statas-robust-option-in-r)
hc3_test = coeftest(model_fixedeffects_9catdocsep, vcov = vcovHC(model_fixedeffects_9catdocsep, "HC3")) 


# -----------------------------------------------------------------------------------------
### Get mean predicted probabilities using MARGINAL EFFECTS PACKAGE https://marginaleffects.com/chapters/predictions.html
# Generate averages including IPW weights
marginal_effects_predictions_counterfactual_agg_weighted_9catdocsep <- avg_predictions(model_fixedeffects_9catdocsep, 
                                                                                       by = exposure_var,
                                                                                       variables = exposure_var,
                                                                                       vcov = 'HC3',
                                                                                       conf_level = 0.95,
                                                                                       wts = weight_var
)

# Export mean predicted probabilities as csv file
if (export_csv == 1) {
  write.csv(marginal_effects_predictions_counterfactual_agg_weighted_9catdocsep, paste("LFS_MarginalEffectsResults_Outcome_", outcome_binary_string, "_LogisticRegression_ProbabilityResults_", exposure_var, ".csv", sep = ""))
}

# ------------------------------------------------------------------------------
# Exposure = NS-SEC 3-cat with doctor separated
# ------------------------------------------------------------------------------
# Estimate Fixed-Effects only Logistic Regression model
# Specify exposure variable
exposure_var = "nssec_familybackground_separatedoctor_3cat"

# Adapted from https://www.statology.org/r-logistic-regression-predict/
# Fit the fixed-effects only logistic regression
model_fixedeffects_3catdocsep <- glm(outcome_binary ~ lfs_year + year_age18_5yrband_agg_doctoronly + SEX + CRY12 + ETHUKEUL + nssec_familybackground_separatedoctor_3cat, 
                                     data=data, 
                                     weights = IPW_ProvidedWeightPlusSelection_winsorised,
                                     family=binomial # logistic regression
)

#view model summary
summary(model_fixedeffects_3catdocsep)

# -----------------------------------------------------------------------------------------
### Get mean predicted probabilities using MARGINAL EFFECTS
marginal_effects_predictions_counterfactual_agg_weighted_3catdocsep <- avg_predictions(model_fixedeffects_3catdocsep, 
                                                                                       by = exposure_var,
                                                                                       variables = exposure_var,
                                                                                       vcov = 'HC3',
                                                                                       conf_level = 0.95,
                                                                                       wts = weight_var
)

# Export mean predicted probabilities as csv file
if (export_csv == 1) {
  write.csv(marginal_effects_predictions_counterfactual_agg_weighted_3catdocsep, paste("LFS_MarginalEffectsResults_Outcome_", outcome_binary_string, "_LogisticRegression_ProbabilityResults_", exposure_var, ".csv", sep = ""))
}


# ------------------------------------------------------------------------------
# Exposure = NS-SEC 3-cat
# ------------------------------------------------------------------------------
# Estimate Fixed-Effects only Logistic Regression model
# Specify exposure variable
exposure_var = "nssec_familybackground_3cat"

# Adapted from https://www.statology.org/r-logistic-regression-predict/
# Fit the fixed-effects only logistic regression
model_fixedeffects_3cat <- glm(outcome_binary ~ lfs_year + year_age18_5yrband_agg_doctoronly + SEX + CRY12 + ETHUKEUL + nssec_familybackground_3cat, 
                               data=data, 
                               weights = IPW_ProvidedWeightPlusSelection_winsorised,
                               family=binomial # logistic regression
)

#view model summary
summary(model_fixedeffects_3cat)

# -----------------------------------------------------------------------------------------
### Get mean predicted probabilities using MARGINAL EFFECTS
marginal_effects_predictions_counterfactual_agg_weighted_3cat <- avg_predictions(model_fixedeffects_3cat, 
                                                                                 by = exposure_var,
                                                                                 variables = exposure_var,
                                                                                 vcov = 'HC3',
                                                                                 conf_level = 0.95,
                                                                                 wts = weight_var
)

# Export mean predicted probabilities as csv file
if (export_csv == 1) {
  write.csv(marginal_effects_predictions_counterfactual_agg_weighted_3cat, paste("LFS_MarginalEffectsResults_Outcome_", outcome_binary_string, "_LogisticRegression_ProbabilityResults_", exposure_var, ".csv", sep = ""))
}


# ------------------------------------------------------------------------------
# Exposure = NS-SEC 9-cat
# ------------------------------------------------------------------------------
# Estimate Fixed-Effects only Logistic Regression model
# Specify exposure variable
exposure_var = "nssec_familybackground_9cat"

# Adapted from https://www.statology.org/r-logistic-regression-predict/
# Fit the fixed-effects only logistic regression
model_fixedeffects_9cat <- glm(outcome_binary ~ lfs_year + year_age18_5yrband_agg_doctoronly + SEX + CRY12 + ETHUKEUL + nssec_familybackground_9cat, 
                               data=data, 
                               weights = IPW_ProvidedWeightPlusSelection_winsorised,
                               family=binomial # logistic regression
)

#view model summary
summary(model_fixedeffects_9cat)

# -----------------------------------------------------------------------------------------
### Get mean predicted probabilities using MARGINAL EFFECTS 
marginal_effects_predictions_counterfactual_agg_weighted_9cat <- avg_predictions(model_fixedeffects_9cat, 
                                                                                 by = exposure_var,
                                                                                 variables = exposure_var,
                                                                                 vcov = 'HC3',
                                                                                 conf_level = 0.95,
                                                                                 wts = weight_var
)

# Export mean predicted probabilities as csv file
if (export_csv == 1) {
  write.csv(marginal_effects_predictions_counterfactual_agg_weighted_9cat, paste("LFS_MarginalEffectsResults_Outcome_", outcome_binary_string, "_LogisticRegression_ProbabilityResults_", exposure_var, ".csv", sep = ""))
}


# ------------------------------------------------------------------------------
# Exposure = Main earner SOC2020 occupation
# ------------------------------------------------------------------------------
# Estimate Fixed-Effects only Logistic Regression model
# Specify exposure variable
exposure_var = "SOC_occupation_familyback_separatedoctor"

# Adapted from https://www.statology.org/r-logistic-regression-predict/
# Fit the fixed-effects only regression
model_fixedeffects_SOC2020 <- glm(outcome_binary ~ lfs_year + year_age18_5yrband_agg_doctoronly + SEX + countryofbirth_binary + ethnicgroup_category + SOC_occupation_familyback_separatedoctor, 
                                  data=data, 
                                  weights = IPW_ProvidedWeightPlusSelection_winsorised,
                                  family=binomial # logistic regression
)

#view model summary
summary(model_fixedeffects_SOC2020)

# -----------------------------------------------------------------------------------------
### Get mean predicted probabilities using MARGINAL EFFECTS

# Too big to do in one go, so split into groups of 10 exposure category values
SOC_all <- as.list(levels(data$SOC_occupation_familyback_separatedoctor))
SOC_part1 <- as.list(levels(data$SOC_occupation_familyback_separatedoctor)[1:20])
SOC_part2 <- as.list(levels(data$SOC_occupation_familyback_separatedoctor)[21:40])
SOC_part3 <- as.list(levels(data$SOC_occupation_familyback_separatedoctor)[41:60])
SOC_part4 <- as.list(levels(data$SOC_occupation_familyback_separatedoctor)[61:80])
SOC_part5 <- as.list(levels(data$SOC_occupation_familyback_separatedoctor)[81:100])
SOC_part6 <- as.list(levels(data$SOC_occupation_familyback_separatedoctor)[101:106])

for (n in 1:length(SOC_all)) {
  print (n)
  # Get mean predicted probabilities using MARGINAL EFFECTS 
  marginal_effects_predictions_counterfactual_agg_weighted_SOC2020_n <- avg_predictions(model_fixedeffects_SOC2020, 
                                                                                        by = exposure_var,
                                                                                        variables = list(SOC_occupation_familyback_separatedoctor = SOC_all[n]),
                                                                                        vcov = 'HC3',
                                                                                        conf_level = 0.95,
                                                                                        wts = weight_var
  )
  
  # Export individual result (in case of failure part way through)
  if (export_csv == 1) {
    write.csv(marginal_effects_predictions_counterfactual_agg_weighted_SOC2020_n, paste("LFS_MarginalEffectsResults_Outcome_", outcome_binary_string, "_LogisticRegression_ProbabilityResults_", exposure_var, "_", n, ".csv", sep = ""))
  }
  
  # Append individual results together
  if (n == 1) {
    marginal_effects_predictions_counterfactual_agg_weighted_SOC2020 <- marginal_effects_predictions_counterfactual_agg_weighted_SOC2020_n
  } else {marginal_effects_predictions_counterfactual_agg_weighted_SOC2020 <- rbind(marginal_effects_predictions_counterfactual_agg_weighted_SOC2020, marginal_effects_predictions_counterfactual_agg_weighted_SOC2020_n)
  
  }
}


# Export mean predicted probabilities as csv file
if (export_csv == 1) {
  write.csv(marginal_effects_predictions_counterfactual_agg_weighted_SOC2020, paste("LFS_MarginalEffectsResults_Outcome_", outcome_binary_string, "_LogisticRegression_ProbabilityResults_", exposure_var, "_all.csv", sep = ""))
}


