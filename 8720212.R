
library(scales)
library(dplyr)
library(tidyverse)
library(caret)
library(MASS)
#loading necessary packages
library(readr)
PitchData_v2 <- read_csv("Desktop/PitchData_v2.csv")
#importing data from desktop
library(ggplot2)
target <- c("Fastball")
#filtering data using dplyr so that only fastballs are shown
test <- filter(PitchData_v2, PitchType %in% target)
histogram(test$ReleaseSpeed, xlab = "Release Speed")
#looking for outlier fastballs to be removed from data, fastballs less than 80 are removed from the data
#due to low occurences -I hypothesize that these fastballs either come from position players or are misclassified pitches given the lowest average fastball velocity in the MLB was 81.4mph by R.A. Dickey in 2016 (closest data I could find)

filt <- PitchData_v2 %>%
  filter(ReleaseSpeed >= 80) #filtering out said fastballs
filt1 <- na.omit(filt)
dataSet <- filt1 %>% filter(PitchType %in% target) #creating dataset with filtered pitch types and velocities
attach(dataSet)
dataSet$sameSide [BatSide == PitchHand] <- 1

dataSet$sameSide [BatSide != PitchHand] <- 0

#baseball logic dictates righty/righty or lefty/lefty matchups are difficult for the batter, so I generate a categorical variable that is one in same side matchups and 0 in opposite matchups to test this conventional wisdom

dataSet$sStrike [PitchResult == "SwingingStrike" | PitchResult == "SwingPitchout" | PitchResult == "SwingStrikeBlk"] <- 1
dataSet$sStrike [PitchResult != "SwingingStrike" & PitchResult != "SwingPitchout" & PitchResult != "SwingStrikeBlk"] <- 0

#creating result variable sStrike, since there are multiple outcomes that constitute a swinging strike (and multiple that do not), this makes one variable that is a 1 in the case of a swinging strike and a 0 otherwise
typeof(dataSet$SpinRate) #checking the type of variable spin rate and changing it from a character to numeric values otherwise the model will not run correctly
dataSet$SpinRate <- as.numeric(as.character(dataSet$SpinRate))
#splitting data into two parts, 60% of the data for model training and 40% of the data for model testing

intrain <- createDataPartition(y=dataSet$sStrike, p =0.60, list = FALSE)
cleanedData <- na.omit(dataSet)
plot1 <- ggplot(dataSet, aes(VertApproachAngle, ReleaseSpeed, colour=sStrike)) + geom_point(alpha = 0.1)
training <- cleanedData[intrain,]
testing <- cleanedData[-intrain,]

#for my model, since the outcome is binary (either the result of a pitch is a swinging strike or it's not), I decided to use logistic regression to build out the model

myModel <- glm(sStrike ~ ReleaseSpeed + SpinAxis + PitchOfPA + Strikes + SpinRate + HorzBreakPFX + VertBreakPFX + VertApproachAngle + HorzApproachAngle + ReleaseSide + Extension + sameSide, data = training, family = binomial(link = logit)) %>% stepAIC(trace = FALSE)
summary(myModel)

#after building the model using stepwise regression I was inclined to remove pitchOfPA due to its low significance (less than 0.05), so I built a new model without it

newModel <- glm(sStrike ~ ReleaseSpeed + Strikes + SpinAxis + SpinRate + HorzBreakPFX + VertBreakPFX + VertApproachAngle + HorzApproachAngle + ReleaseSide + Extension + sameSide, data = training, family = binomial(link = logit)) %>% stepAIC(trace = FALSE)
summary(newModel)

#as you can see, deviance is reduced from about 193k to 116k
exp(coef(newModel))

#exponentiating coefficients for easier interpretability because logisitic regression is output in the form of log odds ratio
probabilities <- newModel %>% predict(testing, type = "response")

#creating predicted probabilities for swinging strikes using the partition of data set aside for model testing

testing$probabilities <- probabilities
testing <- na.omit(testing)

#adding probabilites column to testing dataset and removing NA values
#plotting VertApproachAngle (the angle at which the ball approaches the strike zone) against my predicted probabilities from the model
#one of the most surprising results from the model, the flatter you throw a ball (the less steeply a ball enters the strike zone downwards) the more likely a swinging strike is to occur, illustrated in this visualization only including the subset of our data that was swinging strikes 

vertPlot<-ggplot(testing, aes(VertApproachAngle, probabilities, colour = sStrike)) + geom_point(alpha = 0.1)
vertPlot %+% subset(testing, sStrike %in% c(1))
mean(testing$probabilities) 

speedPlot <- ggplot(testing, aes(ReleaseSpeed, probabilities, colour = sameSide)) + geom_point(alpha = 0.1)
speedPlot %+% subset(testing, sameSide %in% c(0))
#in this plot, we observe the subset of Release speed vs. probabilities (of being a swinging strike) that were Righty vs. Lefty matchups (the batter and pitcher were on opposite sides)
speedPlot1 <- speedPlot %+% subset(testing, sameSide %in% c(1))
speedPlot1 + scale_colour_gradient(low = "#D55E00", high = "#D55E00",space = "Lab", na.value = "grey50", guide = "colourbar",aesthetics = "colour" )
#in this plot, we observe the other subset of batters, those that were on the same side of the pitcher IE righty righty or lefty lefty

#another striking takeaway from my model is the differing groups of batters.  Velocity causes a much more noticable increase in swinging strike probability
#when the batters are on the same sides as the pitchers as opposed to whether they were on the opposite side.  
#The magnitude of the effect of Velocity is highly dependent on the batter pitcher handedness matchup 

#calculating mean of probabilities
#creating confusion matrix to evaluate model using predicted probabilities

data.pred <- rep(0, dim(testing)[1])
data.pred[probabilities > 0.06881322] = 1

#setting cut off value to mean of predicted probabilities, everything above this value will be considered a swinging strike and everything below will be considered not a swinging strike
data.pred <- na.omit(data.pred)
table(data.pred, testing$sStrike)

#outputting confusion matrix
mean(data.pred == testing$sStrike)

#seeing the results of our models predictions, using this we can see that the model is correct around 61% of the time, doing much better than random guessing at 50%

