## !/user/bin/env RStudio 1.1.423
## -*- coding: utf-8 -*-
## KNN Model

# 加载扩展包：
library("dplyr")
library('caret')
library('magrittr')

# 清空无关内存
rm(list = ls())
gc()

# 数据导入代码（数据导入、特征标准化、训练集与测试机划分）
Data_Input <- function(file_path = "D:/R/File/iris.csv",p = .75){
    data = read.csv(file_path,stringsAsFactors = FALSE,check.names = FALSE)
    names(data) <- c('sepal_length','sepal_width','petal_length','petal_width','class')
    data[,-ncol(data)] <- scale(data[,-ncol(data)])
    data['class_c'] =  as.numeric(as.factor(data$class))
    x = data[,1:(ncol(data)-2)];y =  data$class_c
    samples = sample(nrow(data),p*nrow(data))
    train_data  =  x[samples,1:(ncol(data)-2)];train_target = y[samples]
    test_data   =  data[-samples,1:(ncol(data)-2)];test_target = y[-samples]
    return(
        list(
            data = data,
            train_data = train_data, 
            test_data = test_data,
            train_target = train_target,
            test_target = test_target
            )
        )
 }


# 构建KNN分类器：
kNN_Classify <-function(test_data,test_target,train_data,train_target,k){
    # step 1: 计算距离
    centr_matrix = unlist(rep(test_data,time = nrow(train_data)),use.names = FALSE)  %>%
    matrix(byrow = TRUE,ncol = 4) 
    diff = as.matrix(train_data) - centr_matrix  
    squaredDist = apply(diff^2,1,sum) 
    distance = as.numeric(squaredDist ^ 0.5)

    # step 2: 对距离排序
    sortedDistIndices = rank(distance)
    classCount = c()
    for (i in 1:k){
        # step 3: 选择k个最近邻居
        target_sort = train_target[sortedDistIndices == i]
        classCount = c(classCount,target_sort)
    }
    # step 4: 分类统计并返回频数最高的类
    Max_count = plyr::count(classCount) %>% arrange(-freq) %>%.[1,1]
    return (Max_count)
}

# 测试集、验证集收集：
data_source  <- Data_Input()
train_data   <- data_source$train_data
test_data  <- data_source$test_data
train_data   <- data_source$train_data
train_target <- data_source$train_target

# 单样本测试（需运行第一个数据导入代码）
kNN_Classify(
    test_data = test_data ,
    test_target = test_target,
    train_data = train_data,
    train_target = train_target,
    k = 5
    )


# 运行分类器代码：
datingClassTest <- function(test_data,train_data,train_target,test_target,k = 5){
    m = nrow(test_data)
    w = ncol(test_data)
    errorCount = 0.0
    test_predict = c()
    for (i in 1:m){
        classifierResult =  kNN_Classify(
            test_data    = test_data[i,],
            train_data   = train_data,
            train_target = train_target,
            k = k
            )
        if (classifierResult != test_target[i]){
            errorCount =  errorCount + 1.0
        }
        test_predict = c(test_predict,classifierResult)
    }
    test_data[['test_predict']] = test_predict
    test_data[['class']] = test_target
    target_names = c('setosa', 'versicolor', 'virginica')
    print(confusionMatrix(factor(test_predict,labels = target_names),factor(test_target,labels = target_names)))
    confusion_matrix = table(test_predict,test_target) 
    dimnames(confusion_matrix) <- list(target_names,target_names)
    cat(sprintf("in datingClassTest,the total error rate is: %f",errorCount/m),sep = '\n')
    cat(sprintf('in datingClassTest,errorCount:%d',errorCount),sep = '\n')
    return (list(test_data = test_data,confusion_matrix = confusion_matrix))
}


result <- datingClassTest(
    test_data = test_data,
    train_data = train_data,
    train_target = train_target,
    test_target = test_target
    )

result$test_data
result$confusion_matrix
