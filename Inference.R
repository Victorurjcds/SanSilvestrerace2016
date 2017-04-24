
##########################################
# San Silvestre race in 2016
##########################################

# clear workspace
rm(list = ls())

# define work directory
setwd('C:/Users/Victor/Documents/Ambito_profesional/proyectos/s_s_race')



####################################################################################
# Install packages
install.packages('nortest')
library('nortest')
install.packages('MASS')
library('MASS')
install.packages('ggplot2')
library('ggplot2')
install.packages('mixtools')
library('mixtools')



####################################################################################
# import a csv file
data <- read.csv('resultados_popular.csv',sep=';')
data <- data[,1:4]
data$Tiempo <- strptime(data$Tiempo, format="%H:%M:%S")
data$Tiempo <- data$Tiempo$hour*3600+data$Tiempo$min*60+data$Tiempo$sec

# For doing Kolmogorov-Smirnov: not repited values. Handle this:
for (i in 1:length(data$Tiempo)) {
  if (length(which(data$Tiempo %in% data$Tiempo[i]))>1) {
      decimal <- (1:length(which(data$Tiempo %in% data$Tiempo[i])))*1/(length(which(data$Tiempo %in% data$Tiempo[i]))+1)
      data$Tiempo[which(data$Tiempo %in% data$Tiempo[i])] <- data$Tiempo[which(data$Tiempo %in% data$Tiempo[i])]+decimal
  }
}
desv <- sd(data$Tiempo)
data$Tiempo <- data$Tiempo/desv



####################################################################################
# exploring data
data$Tiempo
plot(data$Tiempo)
hist(data$Tiempo, breaks=sqrt(length(data$Tiempo)))   # Parece normal.

# H0: los datos proceden de una distribución normal
# H1: los datos no proceden de una distribución normal
n_test <- lillie.test(data$Tiempo)
if (n_test[2]<0.1) {
  print('Tiempo variable is not represented as a Normal distribution.')
}

# H0: Dataset follows as a Gamma distribution.
# H1: Dataset doesn't follow as a Gamma distribution
fit.params <- fitdistr(data$Tiempo,'gamma') 
temp <- rgamma(1000000, shape=fit.params$estimate[1], rate=fit.params$estimate[2])
plot(density(temp, n=length(data$Tiempo)), main='Gamma distribution vs San Silvestre')
lines(density(data$Tiempo, n=length(data$Tiempo)), col='red')
plot(ecdf(temp), main='CDF Gamma distributions vs San Silvestre')
lines(ecdf(data$Tiempo), col='red')
ks.test(data$Tiempo, y=temp)



# Probability runner ends racer at time shorter than 'x' minutes:
time1 <- 45
time2 <- 57
time3 <- 31
time4 <- 72

prob1 <- pgamma((time1*60)/desv, shape=fit.params$estimate[1], rate=fit.params$estimate[2])*100
sprintf('Probability for time shorter than %i minutes is: %f%%.', time1, prob1)
prob2 <- pgamma((time2*60)/desv, shape=fit.params$estimate[1], rate=fit.params$estimate[2])*100
sprintf('Probability for time shorter than %i minutes is: %f%%.', time2, prob2)
prob3 <- pgamma((time3*60)/desv, shape=fit.params$estimate[1], rate=fit.params$estimate[2])*100
sprintf('Probability for time shorter than %i minutes is: %f%%.', time3, prob3)
prob4 <- pgamma((time4*60)/desv, shape=fit.params$estimate[1], rate=fit.params$estimate[2])*100
sprintf('Probability for time shorter than %i minutes is: %f%%.', time4, prob4)







