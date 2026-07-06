

# CKD / DKD MULTI-LAYER ML PIPELINE (CLEAN VERSION)

rm(list = ls())

# 1. LIBRARIES
library(readxl)

library(tidyverse)

library(caret)

library(xgboost)

library(pROC)
# 2. LOAD DATA
data_path <- "C:/Users/playm/Downloads/Book1.xlsx"

df <- read_excel(data_path, sheet = 1)
# 3. BASIC CLEANING
df <- df %>%

  mutate(across(where(is.numeric),

                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 4. DEFINE OUTCOME VARIABLE
df$Outcome <- as.numeric(df$Group)
# 5. FIX BASIC CODING (ENSURE NUMERIC)
idence <- as.numeric(df$Residence)

df$Gender    <- as.numeric(df$Gender)

df$Diabetes  <- as.numeric(df$Daibetices)

df$BP        <- as.numeric(df$Bp)

# 6. FEATURE ENGINEERING (MULTI-LAYER SYSTEM)
# LAYER 1: Genetic Risk Score

df$GRS <- scale(df$rs1800795 + df$rs564481)
# LAYER 2: Metabolic / Inflammation Proxy

df$InflammationScore <- scale(

  scale(df$BMI) +

    scale(df$Age) +

    scale(df$Diabetes)

)
# LAYER 3: Clinical Kidney Risk Proxy

df$KidneyRiskScore <- scale(

  scale(df$Urea) +

    scale(df$Creatinine) +

    scale(df$Uric.acid) +

    scale(df$IPTH)

)



# LAYER 4: Electrolyte / Hematology Proxy (if exist in file)
df$SystemicScore <- scale(

  scale(df$WBC) +

    scale(df$Platelet) +

    scale(df$HB)

)

# 7. FINAL MODEL DATASET
model_df <- df %>%

  select(

    GRS,

    InflammationScore,

    KidneyRiskScore,

    SystemicScore,

    Outcome

  )
# 8. TRAIN / TEST SPLIT
set.seed(123)
train_index <- createDataPartition(model_df$Outcome, p = 0.8, list = FALSE)
train <- model_df[train_index, ]

test  <- model_df[-train_index, ]



x_train <- as.matrix(train %>% select(-Outcome))

y_train <- train$Outcome



x_test <- as.matrix(test %>% select(-Outcome))

y_test <- test$Outcome


# 9. XGBOOST MODEL
model <- xgboost(

  data = x_train,

  label = y_train,

  nrounds = 300,

  objective = "binary:logistic",

  eval_metric = "auc",
  verbose = 0

)
# 10. PREDICTION
pred <- predict(model, x_test)
# 11. EVALUATION
roc_obj <- roc(y_test, pred)
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))
plot(roc_obj, col = "blue", main = "ROC Curve")


# 12. FEATURE IMPORTANCE
importance <- xgb.importance(colnames(x_train), model)
print(importance)

xgb.plot.importance(importance)



# 13. SAVE MODEL
xgb.save(model, "FINAL_DKD_MODEL.xgb")

#########################################################

