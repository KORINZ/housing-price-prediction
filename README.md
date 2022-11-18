<a name="readme-top"></a>

[![LinkedIn][linkedin-shield]][linkedin-url]
<!-- PROJECT TITLE -->
<div align="center">
<h1 align="center"> Housing Price Predication with Machine Learning </h1>
</div>

Using California census data (1990) to build a model of housing prices. The data includes 
longitude, latitude, housing median age, total rooms, total bedrooms, population, households, median income, median house value, and ocean proximity for each district. The model should
learn from this data and be able to predict the median housing price in any district given
all the other metrics mentioned above.

## Project Checklist

### 1. Establishing the problem frame

- The model outputs (predictions of median housing price for each district) should be able to be fed into another machine learning system.
- The current solution involves manually calculate the median housing prices with complex formula; it is
time-consuming and the estimates are not ideal (off by ~30%).
- Since the inputs are labeled with expected outputs, this is a **supervised learning** task. Moreover, it
is a **regression problem** since the expected outputs are values.
- The model will be trained with **batch learning** since there is no constant update 
of the data, and the data size considerably small.
- The **root-mean-square error (RMSE)** will be used to evaluate the performance of the regression problem since the amount of outlier data are not significant.

### 2. Acquiring the data

- The data of this project is available at: https://github.com/ageron/data/raw/main/housing.tgz.

### 3. Gaining insight from the data

- 

### 4. Preprocessing the data

- Creating training and test sets.

### 5. Choosing a model

-

### 6. Fine-tuning the model

-

### 7. Presenting the solution

-

### 8. Launching, monitoring, and maintaining the system

-


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/colin-z/
