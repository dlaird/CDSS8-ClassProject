# Practical Machine Learning - Course Project
David Laird  
Saturday, September 20, 2014  

## Introduction
This analysis is performed as part of the requirements of the Practical Machine Learning course of the Coursera/Johns Hopkins University Data Science Specialization.  A data set is provided containing observations on body movement taken from subjects while doing exercise, along with a 5-level categorical determination of how correctly the exercise was done.  This objective of this exercise is to create an accurate predictive model which can be applied to 20 individual test cases which will be submitted for grading separately.  More information about the study originating the data can be found at:

*http://groupware.les.inf.puc-rio.br/har*

## Loading Data 
The training data set and the quiz data set provided by the assignment are loaded.  All model fitting analysis will be performed on the training data, but the quiz data will be used to help in column selection to make sure that the model is built only on variables that are available for making predictions.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
d = read.csv("pml-training.csv")
qz = read.csv("pml-testing.csv")
```

## Exploratory Data Analysis
The training set is explored using the `str()`, `head()` and `summary()` functions and is found to contain the response column (_classe_) plus 159 predictor columns made up of a combination of numerical variables and factor variables.  The response takes on one of five categorical values (A through E).  The predictor variables are assumed to be mainly measurements of motion during activity, but their precise definitions and metric units are not known nor were they found on any data dictionary at the source website provided.  The quiz set has the same structure except that the response variable is replaced with a problem identifier.

## Column and Row Selection
**Column Selection:**  The quiz data is inspected visually and many columns are found to contain no predictive information.  One column contain all "no"s and rest contain only NAs.  Since the model will not be able to use these columns for making predictions, they will be removed from the training data set.  The following step allows for confirmation that the identified columns do not contain useful data, and then reduces the number of explanatory columns in the training set from 159 to 58.

```r
emptyColumns = c(6,12:36,50:59,69:83,87:101,103:112,125:139,141:150)
# the next line simply serves as a check that all removed columns contain no useful information
qzRemove = qz[emptyColumns] 
d = d[,-emptyColumns]
```
Some of the remaining columns contain factor variables, and since factor variables cannot be used in principal component analysis, they will also be removed, with the exception of the response variable *classe*.  The response variable is the last column in the set, which is why the for loop subtracts one from the dimension of the dataset.

```r
fctrs = c()
for (i in 1:dim(d)[2]-1) {
  if (is.factor(d[,i])){fctrs = c(fctrs,i)}
}
d = d[,-fctrs]
```
The last step in column selection is to remove columns that contain noise.  Specifically the first column, which is just a row count variable, is strongly correlated in the original dataset with the response variable since the data is presented sorted by response.  Two time-stamp columns will also be removed since any influence they will only add noise to the model selection.

```r
noise = c(1,2,3)
d = d[,-noise]
```

**Row Selection:**  The training set consists of 19,622 observation.  This is considered enough to subdivide into a training set to use for model fitting and a validation set to test the model fit and to estimate the out-of-sample error rate.  In addition,  three other test sets of only 22 observations each are created to replicate the quiz data set size to get additional estimates of quiz performance.   Since the available data set is significantly larger than those used in lecture examples, a slightly larger proportion will be used for training with some assurance that the remaining set will be large enough to provide reasonable validation.  Specifically, 80% of the data will be used for training and 20% for validation.

```r
inTst1 = createDataPartition(y=d$classe, p=0.001019264, list=FALSE)
dtst1 = d[inTst1,]
d = d[-inTst1,]
inTst2 = createDataPartition(y=d$classe, p=0.001019264, list=FALSE)
dtst2 = d[inTst2,]
d = d[-inTst2,]
inTst3 = createDataPartition(y=d$classe, p=0.001019264, list=FALSE)
dtst3 = d[inTst3,]
d = d[-inTst3,]
inTrn = createDataPartition(y=d$classe, p=0.8, list=FALSE)
dtrn = d[inTrn,]
dval = d[-inTrn,]
```

## Preprocessing
Preprocessing with principal component analysis (PCA) has the objective of reducing the large number of explanatory variables to a smaller number while retaining a large degree of predictive information contained in the entire data.  The result is a new data set with only numerical columns and no missing values. 

The first step is to remove the response factor variable from the explanatory variables for easier preprocessing of the latter.

```r
dtrnResponse = dtrn[54]
dtrn = dtrn[-54]
```
Since PCA which does not allow for missing values, the next step in preprocessing is to replace missing values with imputed values using the `medianImpute` method.  

```r
dtrnPre = preProcess(dtrn,
                     method = "medianImpute")
dtrni = predict(dtrnPre,
                newdata=dtrn)
```
The nature of the data is not well understood by the author but as a precaution all columns are normalized by centering and scaling.  This produces a new data set with scaled columns. 

```r
dtrnPre = preProcess(dtrni,
                     method = c("center","scale"))
dtrnic = predict(dtrnPre,
                 newdata = dtrni)
```
Finally, PCA is applied to reduce the columns to a more computationally manageable number.  The PCA caret::preProcess object created in this step will be applied to the validation data to similarly transform the columns on that data set into a form that can be consumed by the model fit to the training data. 

```r
dtrnPre = preProcess(dtrnic,
                     method = "pca",
                     thresh = .95)
dtrnicp = predict(dtrnPre,
                  dtrnic)
```
At this point the numerical explanatory columns are reduced from 54 to 25 while retaining 95% of the variability contained in the whole data set.  The result is the final training data set that will be used for model fitting in the next step.

## Model Building
A random forest method is used to estimate a predictive model based on the given and preprocessed training data.  Random Forest was selected because it is known to produce good results with this type of classification prediction and because of the many models available it is the one that is best understood by the author.  The model is fit with _classe_ as the response variable and all of the principal components calculated above as predictors.

```r
rfFit = train(dtrnResponse$classe ~ .,
              data = dtrnicp,
              method = "rf")
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

**Cross Validation:** Cross-validation is performed by the caret::train function using the default settings for the "rf" (random forest) model.  This results in standard bootstrapping with replacement using 25 resamples. 

**Predicted Values:** Once the model is fit it is run on the training data and the predicted values are compared to the actual training set response variable.  Unsurprisingly, the fit is very good.  (Note:  In fact, the fit is perfect.  What is surprising to the author, however, is that preliminary analyses with much less exact estimation of principal components (`pcaComp = 2`) also yielded perfect fits on the training data.  As expected, these versions did significantly worse in their out-of-sample error rate.) 

```r
predTrn = predict(rfFit,
                  dtrnicp)
confusionMatrix(predTrn,dtrnResponse$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4450    0    0    0    0
##          B    0 3028    0    0    0
##          C    0    0 2728    0    0
##          D    0    0    0 2564    0
##          E    0    0    0    0 2876
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

## Validation and Test Sample Processing
The model fit on training data is applied to validation data to estimate an out-of-sample error rate.  First, the validation data is pre-processed in the same way that the training data was pre-processed. The response variable is separated, missing values are imputed, and each column is centered and scaled.  The pre-processed data is applied to the PCA caret::preProcess object created using training data to estimate principal components for the validation data set.  Since this procedure will also be applied to the three quiz-size samples, a function is written for processing all validations test sets.  The function is written with the option of outputting the actual predictions or the confusion matrix.

```r
predictionFunction = function(dataName,out="cm"){
  response = dataName[54]
  predictors = dataName[-54]
  ppObj = preProcess(predictors,
                     method="medianImpute")
  predictorsImputed = predict(ppObj,
                              newdata=predictors)
  ppObj = preProcess(predictorsImputed,
                     method = c("center","scale"))
  predictorsImputedScaled = predict(ppObj,
                                   newdata = predictorsImputed)
  predictorsPCA = predict(dtrnPre,
                          predictorsImputedScaled)
  predictions = predict(rfFit,
                        newdata = predictorsPCA)
  if (out=="predictions"){
    predictions
  }
  else {
    confusionMatrix(predictions,response[,1])
  }
}
```
The function is then run on each of the test samples, the large validation set and each of the three quiz-size test sets.

```r
cmval  = predictionFunction(dval)
cmtst1 = predictionFunction(dtst1)
cmtst2 = predictionFunction(dtst2)
cmtst3 = predictionFunction(dtst3)
cmval
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1100    8    0    2    2
##          B    6  745    8    2    0
##          C    2    3  665   27    4
##          D    1    0    8  609    3
##          E    3    1    1    0  710
## 
## Overall Statistics
##                                         
##                Accuracy : 0.979         
##                  95% CI : (0.974, 0.984)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.974         
##  Mcnemar's Test P-Value : 0.01          
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.989    0.984    0.975    0.952    0.987
## Specificity             0.996    0.995    0.989    0.996    0.998
## Pos Pred Value          0.989    0.979    0.949    0.981    0.993
## Neg Pred Value          0.996    0.996    0.995    0.991    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.281    0.191    0.170    0.156    0.182
## Detection Prevalence    0.284    0.195    0.179    0.159    0.183
## Balanced Accuracy       0.992    0.990    0.982    0.974    0.993
```

```r
cmtst1er = cmtst1$overall[1]
cmtst2er = cmtst2$overall[1]
cmtst3er = cmtst3$overall[1]
```
## Estimating the Out-of-Sample Error Rate
The predicted values are compared to the actual validation response variable, and an out-of-sample error rate is determined.  
As shown above table, the out-of-sample error rate is:

Validation Results: 97.9 %.  

The three error rates for the quiz-size samples are:

Test 1: 86.4% 

Test 2: 95.5% 

Test 3: 100% 

## Conclusion
Based on the above results, I expect that out of the 20 test cases to submit for this class project, I will get about 19 right.
