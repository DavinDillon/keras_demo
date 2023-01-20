



library(keras)
library(dplyr)
library(ggplot2)
library(lattice)
library(caret)

df <- read.csv('data.csv')
df$diagnosis<-ifelse(df$diagnosis=="M",1,0)

index <- createDataPartition(df$diagnosis, p=0.8, list=FALSE)

training <- df[index,-c(1,33)]
testing <- df[-index,-c(1,33)]

X_train <- training %>% 
  select(-diagnosis) %>% 
  scale()

y_train <- to_categorical(training$diagnosis)

X_test <- testing %>% 
  select(-diagnosis) %>% 
  scale()

y_test <- to_categorical(testing$diagnosis)

model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = ncol(X_train)) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'sigmoid')

history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model %>% fit(
  X_train, y_train, 
  epochs = 50, 
  batch_size = 32,
  validation_split = 0.3
)

summary(model)
model %>% evaluate(X_test, y_test)

predictions <- model %>% predict(X_test)

View(predictions)
