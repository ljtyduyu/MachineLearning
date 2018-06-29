## !/user/bin/env RStudio 1.1.423
## -*- coding: utf-8 -*-
## K-means Model

library("dplyr")
library('magrittr')
library('ggplot2')

rm(list = ls())
gc()

Data_Input <- function(file_path = "D:/R/File/iris.csv",p = .75){
    data = read.csv(file_path,stringsAsFactors = FALSE,check.names = FALSE)
    names(data) <- c('sepal_length','sepal_width','petal_length','petal_width','class')
    data[,-ncol(data)] <- scale(data[,-ncol(data)])
    x = data[,1:(ncol(data)-2)];y =  data$class_c
    return(
        list(
            data = data,
            train_data = x,
            train_target = y
            )
        )
   }


DistEclud <- function(vecA, vecB){
    diff = rbind(vecA,vecB)
    euclidean = dist(diff) %>% as.numeric()
    return (euclidean)
   }


RandCentre <- function(dataSet, k){  
    n = ncol(dataSet)
    Centres = matrix(nrow = k,ncol = n)
    for(j in 1:n){
        minJ = min(dataSet[,j])
        rangeJ = max(dataSet[,j]) - minJ
        Centres[,j] = (minJ + rangeJ * runif(k)) 
    }
    return (Centres)
} 


Kmeans_Cluster <- function(dataSet,k){
    m = nrow(dataSet)
    ClusterAssment = rep(0,times = 2*m) %>% matrix(nrow = m,ncol = 2)  
    Centres = RandCentre(dataSet,k)
    ClusterChanged = TRUE
    while(ClusterChanged){
        ClusterChanged = FALSE
        for(i in 1:m){
            minDist = Inf
            minIndex = 0
            for(j in 1:k){
                distJI = DistEclud(Centres[j,],dataSet[i,])
                if (distJI < minDist){
                    minDist = distJI
                    minIndex = j
                }
            }
            if(ClusterAssment[i,1] != minIndex){
                  ClusterChanged = TRUE
                  ClusterAssment[i,] = c(minIndex,minDist^2)
            } 
        }
        for(cent in 1:k){
            index = grep(cent,ClusterAssment[,1])
            ptsInClust = dataSet[index,]
            Centres[cent,] = apply(ptsInClust,2,mean)
        }
    print(Centres)
    }
    return (
        list(
            Centres = Centres,
            ClusterAssment = ClusterAssment
            )
        )
}



Show_Result <- function(k){
    data_source = Data_Input()
    dataSet = data_source$train_data
    y = data_source$train_target
    result = Kmeans_Cluster(dataSet,k)
    centroids = result$Centres
    clusterAssment = result$ClusterAssment
    ggplot() +
    geom_point(data = NULL,aes(x = dataSet[,1],y = dataSet[,2],fill = factor(clusterAssment[,1])),shape = 21,colour = 'white',size = 4) +
    geom_point(data = NULL,aes(x= centroids[,1],y = centroids[,2]),fill = 'Red',size = 10,shape = 23) +
    scale_fill_brewer(palette = 'Set1') +
    guides(fill=guide_legend(title=NULL)) +
    theme_void()
   }    
    
Show_Result(3)