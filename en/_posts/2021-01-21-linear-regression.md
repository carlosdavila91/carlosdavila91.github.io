---
title: "Linear Regression with R"
author: "Carlos Dávila Luelmo"
date: "18/1/2021"
output:
  html_document:
    keep_md: true
---

# Introduction

In this blog entry, I'd like to showcase a Linear Regression implementation. I'll do it with a data set about housing and `R`. And I assume that you readers have a base knowledge about statistical inference.

The aim here is to use a Linear Regression model, concentrating more on inference than prediction (although we can perform it anyway)—-this way, we focus more on the relationships between the variables rather than the model's prediction power.

A Linear Regression is a math model that is used to approach the dependence relationships between a dependent variable $Y$, a set of independent variables $X_i$ and a random term $\varepsilon$ (randomness is key, keep it in mind for later). Its equation has this form:

$$
Y_t = \beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + ... + + \beta_p \cdot X_p + \varepsilon
$$

Simply put, the linear regression attempts to **generalize the relation between two or more variables** by minimizing the distance between every data point and the line. That is why we say that it uses the least-squares method to fit the model.

I won’t get into many details, as it’s a widely known concept. You can find multiple excellent explanations and resources out there if you do a quick search for the term at the search engine of your preference.

Another thing to point out about linear regression is that it is a relatively simple model. We can understand and so explain what is doing under the hood when we analyze it. It is one of the most used Machine Learning algorithms for that reason.

The variable that we want to model here is the house's price, as you will see afterward.

The aim is to go through the modeling process. It’ll be more enjoyable for you if I show a hands-on implementation of linear regression.

## Significance level and performance metric

Before We go into the action, I'll spot some assumptions to have them in mind throughout the exercise.

The **significance's level** established for our purpose will be an $\alpha = 0.05$. It is an often confusing term to explain. For me, the best way to put it is this.

In statistics, we are often testing a hypothesis. But we are pessimistic guys, so we usually try the opposite of the hypothesis we want to prove. For example, when we want to check if a variable is normally distributed, we test if it is not normally distributed as our initial hypothesis.

The significance level is the probability that we have to get the result if our initial hypothesis (or null hypothesis) is correct. That means if we have a chance to find this result lower than our $\alpha = 0.05$ --a very low probability--we reject the null hypothesis and vice versa.

I choose this $\alpha$ because it's a convention, but we may have selected another value. The value we chose gives us different type I and II error ratios. Check [this excellent explanation about type I and II errors to learn more](https://www.youtube.com/watch?v=edzQQFNzFjM).

**The performance metric** most used in a Linear Regression is the determination coefficient $R^2$--or its cousin, the adjusted $R^2$. In short, it measures the proportion of variability contained in our data explained by the corresponding model. It ranges between 0 and 1.

Think about the model as a way of generalizing knowledge. If it explains more variability, it will be better predicting when you feed it with a new observation.

With this established, we can commence.

## Creating the working environment

First of all, set the working libraries that you’ll need. Remember to keep it tidy. The fewer dependencies, the more durable and reproducible the work will be.

I’ll use my favorite `R` libraries like the `tidyverse` package, but this is only a possible solution.


```r
library(tidyverse)
library(ggthemes)
library(caret)

theme_set(theme_bw())
```

You'll notice that I use a custom function throughout the post to print tables. It is the following one. In my opinion, `kableExtra` is one of the best doing this. [See the package vignette](https://cran.r-project.org/web/packages/kableExtra/vignettes/awesome_table_in_html.html) if you want to learn more.


```r
library(kableExtra)
library(knitr)

my_kable <- function(table, ...){
  kable(table) %>%
    kable_styling(
      ...,
      bootstrap_options = c("striped", "hover", "condensed", "responsive")
    )
}
```

## The Data set

The data set that I used for this blog post is about housing. It is aggregated at the district level and anonymized.

I did some data cleaning previously to focus on the modeling part in this blog entry.

It contains the following information.


```r
tribble(
  ~Variable, ~Description,
  "price", "Sale price of a property by the owner",
  "resid_area", "Residential area proportion of the city",
  "air_qual", "Air quality of the district",
  "room_num", "Average number of rooms in the district households",
  "age", "Years of the construction",
  "dist1", "Distance to the business center 1",
  "dist2", "Distance to the business center 2",
  "dist3", "Distance to the business center 3",
  "dist4", "Distance to the business center 4",
  "teachers", "Number of teachers per thousand inhabitants",
  "poor_prop", "Poor population proportion in the city",
  "airport", "There is an airport in the city",
  "n_hos_beds", "Number of hospital beds per thousand inhabitants in the city",
  "n_hot_rooms", "Number of hotel bedrooms per thousand inhabitants in the city",
  "waterbody", "What kind of natural water source there is in the city",
  "rainfall", "Average annual rainfall in cubic centimetres",
  "bus_ter", "There is a bus station in the city",
  "parks", "Proportion of land allocated as parks and green areas in the city",
  "Sold", "If the property was sold (1) or not (0)"
) %>%
  my_kable()
```

<table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Variable </th>
   <th style="text-align:left;"> Description </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> price </td>
   <td style="text-align:left;"> Sale price of a property by the owner </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resid_area </td>
   <td style="text-align:left;"> Residential area proportion of the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> air_qual </td>
   <td style="text-align:left;"> Air quality of the district </td>
  </tr>
  <tr>
   <td style="text-align:left;"> room_num </td>
   <td style="text-align:left;"> Average number of rooms in the district households </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:left;"> Years of the construction </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist1 </td>
   <td style="text-align:left;"> Distance to the business center 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist2 </td>
   <td style="text-align:left;"> Distance to the business center 2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist3 </td>
   <td style="text-align:left;"> Distance to the business center 3 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist4 </td>
   <td style="text-align:left;"> Distance to the business center 4 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:left;"> Number of teachers per thousand inhabitants </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:left;"> Poor population proportion in the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> airport </td>
   <td style="text-align:left;"> There is an airport in the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hos_beds </td>
   <td style="text-align:left;"> Number of hospital beds per thousand inhabitants in the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hot_rooms </td>
   <td style="text-align:left;"> Number of hotel bedrooms per thousand inhabitants in the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> waterbody </td>
   <td style="text-align:left;"> What kind of natural water source there is in the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> rainfall </td>
   <td style="text-align:left;"> Average annual rainfall in cubic centimetres </td>
  </tr>
  <tr>
   <td style="text-align:left;"> bus_ter </td>
   <td style="text-align:left;"> There is a bus station in the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> parks </td>
   <td style="text-align:left;"> Proportion of land allocated as parks and green areas in the city </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Sold </td>
   <td style="text-align:left;"> If the property was sold (1) or not (0) </td>
  </tr>
</tbody>
</table>

```r

house <- read_csv("house_clean.csv")
```

# Descriptive Analysis and Visualization

Once you've digested the context, the next step is to take a glimpse at the structure of our data.

I'll show a statistical description of numeric and categorical data. This way, we will characterize the variable types, detect possible missing values, outliers, or variables without or with almost no variance.


```r
num_vars <- house %>% select(where(is.numeric), -Sold) %>% names()
cat_vars <- house %>% select(-all_of(num_vars)) %>% names()

house %>%
  select(all_of(num_vars)) %>%
  map_dfr(summary) %>%
  mutate(across(everything(), as.numeric)) %>%
  add_column(Variable = num_vars, .before = 1) %>%
  my_kable()
```

<table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Variable </th>
   <th style="text-align:right;"> Min. </th>
   <th style="text-align:right;"> 1st Qu. </th>
   <th style="text-align:right;"> Median </th>
   <th style="text-align:right;"> Mean </th>
   <th style="text-align:right;"> 3rd Qu. </th>
   <th style="text-align:right;"> Max. </th>
   <th style="text-align:right;"> NA's </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> price </td>
   <td style="text-align:right;"> 5.00 </td>
   <td style="text-align:right;"> 17.02 </td>
   <td style="text-align:right;"> 21.20 </td>
   <td style="text-align:right;"> 22.53 </td>
   <td style="text-align:right;"> 25.00 </td>
   <td style="text-align:right;"> 50.00 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resid_area </td>
   <td style="text-align:right;"> 30.46 </td>
   <td style="text-align:right;"> 35.19 </td>
   <td style="text-align:right;"> 39.69 </td>
   <td style="text-align:right;"> 41.14 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 57.74 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> air_qual </td>
   <td style="text-align:right;"> 0.38 </td>
   <td style="text-align:right;"> 0.45 </td>
   <td style="text-align:right;"> 0.54 </td>
   <td style="text-align:right;"> 0.55 </td>
   <td style="text-align:right;"> 0.62 </td>
   <td style="text-align:right;"> 0.87 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> room_num </td>
   <td style="text-align:right;"> 3.56 </td>
   <td style="text-align:right;"> 5.89 </td>
   <td style="text-align:right;"> 6.21 </td>
   <td style="text-align:right;"> 6.28 </td>
   <td style="text-align:right;"> 6.62 </td>
   <td style="text-align:right;"> 8.78 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 2.90 </td>
   <td style="text-align:right;"> 45.02 </td>
   <td style="text-align:right;"> 77.50 </td>
   <td style="text-align:right;"> 68.57 </td>
   <td style="text-align:right;"> 94.07 </td>
   <td style="text-align:right;"> 100.00 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist1 </td>
   <td style="text-align:right;"> 1.13 </td>
   <td style="text-align:right;"> 2.27 </td>
   <td style="text-align:right;"> 3.38 </td>
   <td style="text-align:right;"> 3.97 </td>
   <td style="text-align:right;"> 5.37 </td>
   <td style="text-align:right;"> 12.32 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist2 </td>
   <td style="text-align:right;"> 0.92 </td>
   <td style="text-align:right;"> 1.94 </td>
   <td style="text-align:right;"> 3.01 </td>
   <td style="text-align:right;"> 3.63 </td>
   <td style="text-align:right;"> 4.99 </td>
   <td style="text-align:right;"> 11.93 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist3 </td>
   <td style="text-align:right;"> 1.15 </td>
   <td style="text-align:right;"> 2.23 </td>
   <td style="text-align:right;"> 3.38 </td>
   <td style="text-align:right;"> 3.96 </td>
   <td style="text-align:right;"> 5.41 </td>
   <td style="text-align:right;"> 12.32 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist4 </td>
   <td style="text-align:right;"> 0.73 </td>
   <td style="text-align:right;"> 1.94 </td>
   <td style="text-align:right;"> 3.07 </td>
   <td style="text-align:right;"> 3.62 </td>
   <td style="text-align:right;"> 4.99 </td>
   <td style="text-align:right;"> 11.94 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 18.00 </td>
   <td style="text-align:right;"> 19.80 </td>
   <td style="text-align:right;"> 20.95 </td>
   <td style="text-align:right;"> 21.54 </td>
   <td style="text-align:right;"> 22.60 </td>
   <td style="text-align:right;"> 27.40 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> 1.73 </td>
   <td style="text-align:right;"> 6.95 </td>
   <td style="text-align:right;"> 11.36 </td>
   <td style="text-align:right;"> 12.65 </td>
   <td style="text-align:right;"> 16.96 </td>
   <td style="text-align:right;"> 37.97 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hos_beds </td>
   <td style="text-align:right;"> 5.27 </td>
   <td style="text-align:right;"> 6.63 </td>
   <td style="text-align:right;"> 8.00 </td>
   <td style="text-align:right;"> 7.90 </td>
   <td style="text-align:right;"> 9.09 </td>
   <td style="text-align:right;"> 10.88 </td>
   <td style="text-align:right;"> 8 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hot_rooms </td>
   <td style="text-align:right;"> 10.06 </td>
   <td style="text-align:right;"> 100.59 </td>
   <td style="text-align:right;"> 117.28 </td>
   <td style="text-align:right;"> 98.82 </td>
   <td style="text-align:right;"> 141.10 </td>
   <td style="text-align:right;"> 153.90 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> rainfall </td>
   <td style="text-align:right;"> 3.00 </td>
   <td style="text-align:right;"> 28.00 </td>
   <td style="text-align:right;"> 39.00 </td>
   <td style="text-align:right;"> 39.18 </td>
   <td style="text-align:right;"> 50.00 </td>
   <td style="text-align:right;"> 60.00 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> parks </td>
   <td style="text-align:right;"> 0.03 </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0.09 </td>
   <td style="text-align:right;">  </td>
  </tr>
</tbody>
</table>

+ See the eight missing values in `n_hos_beds`? We need to handle these values.
+ All the rest of the features are complete.

Let's do some plots!

First, a scatter matrix that will give us a lot of relevant information. It has:

+ the response variable of our model, `price`, plotted against the rest of the variables;
+ frequency diagrams at the diagonal, so you get a glance of the data distribution;
+ scatter plots at the bottom to see how variables are related to each other, and
+ Pearson’s correlation coefficients at the upper section. [Check this wikipedia article](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) if you are not famliar with it.


```r
my_ggpairs <- function(data, num_vars){
  num_vars <- enquo(num_vars)

  data %>%
    filter(across(everything(), ~!is.na(.x))) %>%
    select(!!num_vars) %>%
    ggpairs(lower = list(continuous = wrap("points", shape = 1, size = .5))) +
    theme(axis.text = element_text(size = 6),
          strip.text.x = element_text(size = rel(.8)),
          strip.text.y = element_text(size = rel(.6)))  
}

house %>% my_ggpairs(price:dist3)
```

<img src="{{ base.url }}/images/blog/linear-regression/unnamed-chunk-5-1.png" width="100%" />

```r
house %>% my_ggpairs(c(price, dist4:parks, -airport, -waterbody, -bus_ter))
```

<img src="{{ base.url }}/images/blog/linear-regression/unnamed-chunk-5-2.png" width="100%" />

Maybe, you have noticed that:

+ Variables like `room_num`, `teachers`, and `poor_prop`, are linearly related to the `price` feature. It can be glimpsed in the scatter plots, and they present a correlation coefficient between -1 and -0.5 and between 0.5 and 1, meaning that those relationships are considered strong (it’s a convention).
+ All `dist` variables are highly correlated with each other. This high level of correlation between explanatory variables is known as _multicollinearity_. We are interested in linear relationships between predictors and the response variable. But, when predictors are correlated to each other, they do not give us relevant information. Commonly, you only take one of the variables that present _multicolinearity_ and leave the others out of the model.
+ `n_hot_rooms` takes a range of discrete values, although it is a numeric variable. It has to do with the data collection process or could be a posterior aggregation. By now, let’s consider it as numeric.
+ Our response variable, `price`, seems to have a distribution close to the normal distribution.
+ Other distributions are skewed to the left or the right. As a follow-up to this exercise, we could try to transform these variables with the appropriate transformation for each case to verify if they would improve our model performance.

Now, It's the turn of the categorical variables.


```r
house %>% transmute(across(all_of(cat_vars), as.factor)) %>% summary() %>% my_kable(full_width = F)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:left;"> airport </th>
   <th style="text-align:left;">          waterbody </th>
   <th style="text-align:left;"> bus_ter </th>
   <th style="text-align:left;"> Sold </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> NO :230 </td>
   <td style="text-align:left;"> Lake          : 97 </td>
   <td style="text-align:left;"> YES:506 </td>
   <td style="text-align:left;"> 0:276 </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> YES:276 </td>
   <td style="text-align:left;"> Lake and River: 71 </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> 1:230 </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> None          :155 </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;"> River         :183 </td>
   <td style="text-align:left;">  </td>
   <td style="text-align:left;">  </td>
  </tr>
</tbody>
</table>

+ There are no missing values among categorical variables; they would have shown up if there were.
+ `bus_ter` has only `YES` values.

We can consider that `bus_ter` is a variable of **zero variance**. It is an uninformative predictor that could ruin the model you want to fit your data.

Another common issue in data sets is **near-zero variance** predictors. It happens all the time. You'll find variables that are **almost a constant** in your data set. It often happens, for example, with categorical data transformed into dummy variables. In general, the preferred approach is keeping all the information possible.

Let's check how we are with the data regarding this matter.


```r
map_dfr(
  house %>% select(all_of(num_vars)),
  ~ list(Sd = sd(.x, na.rm = TRUE), Var = var(.x, na.rm = TRUE))
) %>%
  add_column(Variable = num_vars, .before = 1) %>%
  my_kable(full_width = FALSE)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Variable </th>
   <th style="text-align:right;"> Sd </th>
   <th style="text-align:right;"> Var </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> price </td>
   <td style="text-align:right;"> 9.18 </td>
   <td style="text-align:right;"> 84.31 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resid_area </td>
   <td style="text-align:right;"> 6.86 </td>
   <td style="text-align:right;"> 47.06 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> air_qual </td>
   <td style="text-align:right;"> 0.12 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> room_num </td>
   <td style="text-align:right;"> 0.70 </td>
   <td style="text-align:right;"> 0.49 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 28.15 </td>
   <td style="text-align:right;"> 792.36 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist1 </td>
   <td style="text-align:right;"> 2.11 </td>
   <td style="text-align:right;"> 4.45 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist2 </td>
   <td style="text-align:right;"> 2.11 </td>
   <td style="text-align:right;"> 4.45 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist3 </td>
   <td style="text-align:right;"> 2.12 </td>
   <td style="text-align:right;"> 4.49 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist4 </td>
   <td style="text-align:right;"> 2.10 </td>
   <td style="text-align:right;"> 4.41 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 2.16 </td>
   <td style="text-align:right;"> 4.69 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> 7.14 </td>
   <td style="text-align:right;"> 50.99 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hos_beds </td>
   <td style="text-align:right;"> 1.48 </td>
   <td style="text-align:right;"> 2.18 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hot_rooms </td>
   <td style="text-align:right;"> 51.58 </td>
   <td style="text-align:right;"> 2660.35 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> rainfall </td>
   <td style="text-align:right;"> 12.51 </td>
   <td style="text-align:right;"> 156.59 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> parks </td>
   <td style="text-align:right;"> 0.01 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
</tbody>
</table>

We can now see that `air_qual` has a very low standard deviation and variance, and `parks` shows near-zero variance.

Both quantitative features ranges are very narrow, as we saw at the beginning of this section. In advance, we'd think that those near-zero variances mean that the variables do not hold decisive information. But we have to be sure before removing any information at all of the data set.

One way to double-check it is using the `caret` package. It has the `nearZeroVar()` function to analyze this.


```r
nearZeroVar(house, saveMetrics = TRUE) %>% my_kable(full_width = FALSE)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> freqRatio </th>
   <th style="text-align:right;"> percentUnique </th>
   <th style="text-align:left;"> zeroVar </th>
   <th style="text-align:left;"> nzv </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> price </td>
   <td style="text-align:right;"> 2.0 </td>
   <td style="text-align:right;"> 45.06 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resid_area </td>
   <td style="text-align:right;"> 2.7 </td>
   <td style="text-align:right;"> 4.15 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> air_qual </td>
   <td style="text-align:right;"> 1.1 </td>
   <td style="text-align:right;"> 6.72 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> room_num </td>
   <td style="text-align:right;"> 1.1 </td>
   <td style="text-align:right;"> 8.89 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 10.8 </td>
   <td style="text-align:right;"> 70.36 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist1 </td>
   <td style="text-align:right;"> 1.0 </td>
   <td style="text-align:right;"> 15.61 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist2 </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 69.96 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist3 </td>
   <td style="text-align:right;"> 1.1 </td>
   <td style="text-align:right;"> 16.60 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist4 </td>
   <td style="text-align:right;"> 1.4 </td>
   <td style="text-align:right;"> 69.76 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 2.2 </td>
   <td style="text-align:right;"> 1.98 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> 1.1 </td>
   <td style="text-align:right;"> 42.49 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> airport </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 0.40 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hos_beds </td>
   <td style="text-align:right;"> 1.3 </td>
   <td style="text-align:right;"> 11.26 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hot_rooms </td>
   <td style="text-align:right;"> 1.1 </td>
   <td style="text-align:right;"> 6.13 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> waterbody </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 0.79 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> rainfall </td>
   <td style="text-align:right;"> 1.1 </td>
   <td style="text-align:right;"> 8.30 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> bus_ter </td>
   <td style="text-align:right;"> 0.0 </td>
   <td style="text-align:right;"> 0.20 </td>
   <td style="text-align:left;"> TRUE </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> parks </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 9.88 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Sold </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 0.40 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
</tbody>
</table>

So, we'll only consider the variable `bus_ter` as a zero variance feature.

## Missing values and outliers {#missing-values}

As we saw before, there are missing values only at `n_hos_beds`. You may get to understand better why these are missing by looking at the corresponding rows.


```r
house %>% filter(is.na(n_hos_beds)) %>%
  my_kable() %>%
  scroll_box(width = "100%")
```

<div style="border: 1px solid #ddd; padding: 5px; overflow-x: scroll; width:100%; "><table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> price </th>
   <th style="text-align:right;"> resid_area </th>
   <th style="text-align:right;"> air_qual </th>
   <th style="text-align:right;"> room_num </th>
   <th style="text-align:right;"> age </th>
   <th style="text-align:right;"> dist1 </th>
   <th style="text-align:right;"> dist2 </th>
   <th style="text-align:right;"> dist3 </th>
   <th style="text-align:right;"> dist4 </th>
   <th style="text-align:right;"> teachers </th>
   <th style="text-align:right;"> poor_prop </th>
   <th style="text-align:left;"> airport </th>
   <th style="text-align:right;"> n_hos_beds </th>
   <th style="text-align:right;"> n_hot_rooms </th>
   <th style="text-align:left;"> waterbody </th>
   <th style="text-align:right;"> rainfall </th>
   <th style="text-align:left;"> bus_ter </th>
   <th style="text-align:right;"> parks </th>
   <th style="text-align:right;"> Sold </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 19.7 </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:right;"> 0.58 </td>
   <td style="text-align:right;"> 5.4 </td>
   <td style="text-align:right;"> 73 </td>
   <td style="text-align:right;"> 2.9 </td>
   <td style="text-align:right;"> 2.6 </td>
   <td style="text-align:right;"> 3.0 </td>
   <td style="text-align:right;"> 2.7 </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:right;"> 21.1 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 122 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 44 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 22.6 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.77 </td>
   <td style="text-align:right;"> 6.1 </td>
   <td style="text-align:right;"> 81 </td>
   <td style="text-align:right;"> 2.8 </td>
   <td style="text-align:right;"> 2.4 </td>
   <td style="text-align:right;"> 2.6 </td>
   <td style="text-align:right;"> 2.3 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 12.7 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 142 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 26 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.07 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 25.0 </td>
   <td style="text-align:right;"> 41 </td>
   <td style="text-align:right;"> 0.49 </td>
   <td style="text-align:right;"> 6.2 </td>
   <td style="text-align:right;"> 42 </td>
   <td style="text-align:right;"> 4.2 </td>
   <td style="text-align:right;"> 3.8 </td>
   <td style="text-align:right;"> 4.0 </td>
   <td style="text-align:right;"> 3.9 </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:right;"> 9.5 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 12 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 30 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 18.8 </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:right;"> 0.55 </td>
   <td style="text-align:right;"> 5.9 </td>
   <td style="text-align:right;"> 93 </td>
   <td style="text-align:right;"> 2.5 </td>
   <td style="text-align:right;"> 2.2 </td>
   <td style="text-align:right;"> 2.6 </td>
   <td style="text-align:right;"> 2.1 </td>
   <td style="text-align:right;"> 22 </td>
   <td style="text-align:right;"> 16.2 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 152 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 35 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 19.7 </td>
   <td style="text-align:right;"> 36 </td>
   <td style="text-align:right;"> 0.44 </td>
   <td style="text-align:right;"> 6.0 </td>
   <td style="text-align:right;"> 46 </td>
   <td style="text-align:right;"> 7.1 </td>
   <td style="text-align:right;"> 6.5 </td>
   <td style="text-align:right;"> 7.0 </td>
   <td style="text-align:right;"> 6.6 </td>
   <td style="text-align:right;"> 23 </td>
   <td style="text-align:right;"> 13.4 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 112 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.04 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 33.8 </td>
   <td style="text-align:right;"> 34 </td>
   <td style="text-align:right;"> 0.65 </td>
   <td style="text-align:right;"> 7.2 </td>
   <td style="text-align:right;"> 82 </td>
   <td style="text-align:right;"> 2.1 </td>
   <td style="text-align:right;"> 1.9 </td>
   <td style="text-align:right;"> 2.4 </td>
   <td style="text-align:right;"> 2.0 </td>
   <td style="text-align:right;"> 27 </td>
   <td style="text-align:right;"> 9.6 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 113 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.07 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 7.5 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.68 </td>
   <td style="text-align:right;"> 6.8 </td>
   <td style="text-align:right;"> 91 </td>
   <td style="text-align:right;"> 1.9 </td>
   <td style="text-align:right;"> 1.5 </td>
   <td style="text-align:right;"> 2.0 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 25.8 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 10 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 35 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.3 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.69 </td>
   <td style="text-align:right;"> 5.3 </td>
   <td style="text-align:right;"> 96 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 1.4 </td>
   <td style="text-align:right;"> 1.9 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 151 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.07 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table></div>

No other value in the rest of the features raises suspicion.

Let's replace the missing values in the `n_hos_beds` with the median of the variable. It is just one valid approach, which I consider preferable because you do not lose information. Besides, there are just a few missing values.


```r
house_complete <- house %>%
  mutate(n_hos_beds = case_when(
    is.na(n_hos_beds) ~ median(n_hos_beds, na.rm = TRUE),
    TRUE ~ n_hos_beds
  ))
```

Outliers are a pain in the neck when you want to fit a certain type of model. Linear Regression is one of them.

A nice tool to detect outliers are box plots.


```r
house_complete %>%
  select(all_of(num_vars)) %>%
  pivot_longer(names(.), names_to = "variable", values_to = "valor") %>% ggplot(aes(variable, valor)) +
  geom_boxplot(outlier.shape = 1) +
  facet_wrap(variable ~ ., scales = "free") +
  labs(x = NULL) +
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(size = 8))
```

![]({{ base.url }}/images/blog/linear-regression/unnamed-chunk-11-1.png)<!-- -->

Many variables present values out of the box plot whiskers. In most cases, there is no clear boundary to determine which points are outliers. They form a continuum.

There is just one case where we can be sure that the values are outliers in the `n_hot_rooms` variable. A set of observations are far from the rest.

Those cases are 125. They are many. It could be due to two reasons: they are measurement errors, or those are cities with a meager amount of hotels, as being non-touristic places.

Let's show them with the other variables with values outside of the box plot whiskers.


```r
house_complete %>%
  filter(n_hot_rooms < 40) %>%
  head() %>%
  my_kable() %>%
  scroll_box(width = "100%")
```

<div style="border: 1px solid #ddd; padding: 5px; overflow-x: scroll; width:100%; "><table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> price </th>
   <th style="text-align:right;"> resid_area </th>
   <th style="text-align:right;"> air_qual </th>
   <th style="text-align:right;"> room_num </th>
   <th style="text-align:right;"> age </th>
   <th style="text-align:right;"> dist1 </th>
   <th style="text-align:right;"> dist2 </th>
   <th style="text-align:right;"> dist3 </th>
   <th style="text-align:right;"> dist4 </th>
   <th style="text-align:right;"> teachers </th>
   <th style="text-align:right;"> poor_prop </th>
   <th style="text-align:left;"> airport </th>
   <th style="text-align:right;"> n_hos_beds </th>
   <th style="text-align:right;"> n_hot_rooms </th>
   <th style="text-align:left;"> waterbody </th>
   <th style="text-align:right;"> rainfall </th>
   <th style="text-align:left;"> bus_ter </th>
   <th style="text-align:right;"> parks </th>
   <th style="text-align:right;"> Sold </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.69 </td>
   <td style="text-align:right;"> 5.5 </td>
   <td style="text-align:right;"> 100 </td>
   <td style="text-align:right;"> 1.6 </td>
   <td style="text-align:right;"> 1.3 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 1.3 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 30.6 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.3 </td>
   <td style="text-align:right;"> 13 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 26 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.07 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 12 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.61 </td>
   <td style="text-align:right;"> 5.3 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2.3 </td>
   <td style="text-align:right;"> 2.0 </td>
   <td style="text-align:right;"> 2.4 </td>
   <td style="text-align:right;"> 1.7 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 24.9 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.3 </td>
   <td style="text-align:right;"> 15 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 39 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 52 </td>
   <td style="text-align:right;"> 0.62 </td>
   <td style="text-align:right;"> 6.2 </td>
   <td style="text-align:right;"> 94 </td>
   <td style="text-align:right;"> 1.9 </td>
   <td style="text-align:right;"> 1.5 </td>
   <td style="text-align:right;"> 1.9 </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 19 </td>
   <td style="text-align:right;"> 24.2 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 5.7 </td>
   <td style="text-align:right;"> 10 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 28 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 52 </td>
   <td style="text-align:right;"> 0.62 </td>
   <td style="text-align:right;"> 6.4 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 2.0 </td>
   <td style="text-align:right;"> 1.6 </td>
   <td style="text-align:right;"> 1.9 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 19 </td>
   <td style="text-align:right;"> 15.4 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 8.2 </td>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 41 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 19 </td>
   <td style="text-align:right;"> 35 </td>
   <td style="text-align:right;"> 0.52 </td>
   <td style="text-align:right;"> 6.0 </td>
   <td style="text-align:right;"> 45 </td>
   <td style="text-align:right;"> 4.9 </td>
   <td style="text-align:right;"> 4.6 </td>
   <td style="text-align:right;"> 5.0 </td>
   <td style="text-align:right;"> 4.7 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 9.7 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 6.4 </td>
   <td style="text-align:right;"> 11 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 28 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 36 </td>
   <td style="text-align:right;"> 0.50 </td>
   <td style="text-align:right;"> 5.8 </td>
   <td style="text-align:right;"> 61 </td>
   <td style="text-align:right;"> 3.4 </td>
   <td style="text-align:right;"> 3.3 </td>
   <td style="text-align:right;"> 3.6 </td>
   <td style="text-align:right;"> 3.2 </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:right;"> 11.4 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 7.5 </td>
   <td style="text-align:right;"> 15 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 39 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
</tbody>
</table></div>

The rest of the features show values within the inner quantiles.

There are two things that we can do. Drop those observations or replace the outlier values with other (e.g., the mean, a.k.a. the **expected** value or the minimum).

In another context, we may have the chance to get more information about the data set we are dealing with, but we cannot go any further in this case.

I'll consider these outliers as errors in the data collection process and replace them with a central value like the median.

```r
house_prepared <- house_complete %>%
  mutate(n_hot_rooms = case_when(
    n_hot_rooms < 40 ~ median(n_hot_rooms, na.rm = T),
    TRUE             ~ n_hot_rooms
  ))
```

Another approach with outliers and normally distributed data is the `z-score`. It is a useful technique if your variable is normally distributed.

The process consists in:

+ Standardizing the variable by subtracting the mean and dividing each observation with the variable's standard deviation.
+ Leaving out those values observed at more than three standard deviations of the mean (absolute value of _z_ greater than 3), which will be zero after the standardization.

It is the case of `room_num`. Now, I'll show you the observations that are outside these boundaries.


```r
house_prepared %>%
  select(room_num) %>%
  mutate(room_num_z = (room_num - mean(room_num)) / sd(room_num)) %>%
  filter(abs(room_num_z) > 3) %>%
  arrange(room_num_z) %>%
  my_kable(full_width = FALSE)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> room_num </th>
   <th style="text-align:right;"> room_num_z </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 3.6 </td>
   <td style="text-align:right;"> -3.9 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3.9 </td>
   <td style="text-align:right;"> -3.5 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4.1 </td>
   <td style="text-align:right;"> -3.1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4.1 </td>
   <td style="text-align:right;"> -3.1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.4 </td>
   <td style="text-align:right;"> 3.0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.7 </td>
   <td style="text-align:right;"> 3.4 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.7 </td>
   <td style="text-align:right;"> 3.5 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.8 </td>
   <td style="text-align:right;"> 3.5 </td>
  </tr>
</tbody>
</table>

Again, it's a matter of choosing between deleting or replacing those values. I'll replace them with the maximum to preserve some of the variance.


```r
house_prepared <- house_prepared %>%
  mutate(room_num = case_when(
    room_num <= 4.138 | room_num >= 8.398 ~ median(room_num, na.rm = TRUE),
    TRUE ~ room_num ))
```

# Create a test set

For those who may ask, the usual procedure at this point in a Machine Learning project is, before doing anything else, to create the test set. Set aside a part of the data and not touch it until you get a definitive fine-tuned model.

In this case, I'll focus more on the inferential side of the Linear Regression than on the model's predictive power. So I won't split the data into training and testing sets. I'll use it as is.

# Linear Regression model

## Simple Linear Regression

Let's start simple. I'll fit a model with `price` explained by `teachers` and another defined by `poor_prop`.

In `R` we use the formula syntax. It's a very intuitive way of writing your model. You place your target variable on the left-hand side of the formula and the features you want on the right-hand side and split them with a tilde symbol (`target ~ features`).

The coefficients estimated for the first model are these.

```r
lm.simple_1 <- lm(price ~ teachers, data = house_prepared)
lm.simple_1$coefficients %>% my_kable(full_width = FALSE)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> x </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> (Intercept) </td>
   <td style="text-align:right;"> -23.7 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 2.1 </td>
  </tr>
</tbody>
</table>

So, we can represent the model with the following expression.

$$\overline{y} = -23.676 +2.145\cdot teachers$$

The coefficients for the second model.


```r
lm.simple_2 <- lm(price ~ poor_prop, data = house_prepared)
lm.simple_2$coefficients %>% my_kable(full_width = FALSE)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> x </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> (Intercept) </td>
   <td style="text-align:right;"> 34.58 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> -0.95 </td>
  </tr>
</tbody>
</table>

And the model would be the next.

$$\overline{y} = 34.582 -0.953\cdot poor\_prop$$

You get how this works.

### Differences between models

To get a better understanding of the results, let's use the `summary()` function.


```r
summary(lm.simple_1)
```

```
##
## Call:
## lm(formula = price ~ teachers, data = house_prepared)
##
## Residuals:
##     Min      1Q  Median      3Q     Max
## -18.783  -4.774  -0.633   3.147  31.212
##
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  -23.676      3.529   -6.71  5.3e-11 ***
## teachers       2.145      0.163   13.16  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## Residual standard error: 7.9 on 504 degrees of freedom
## Multiple R-squared:  0.256,	Adjusted R-squared:  0.254
## F-statistic:  173 on 1 and 504 DF,  p-value: <2e-16
```

```r
summary(lm.simple_2)
```

```
##
## Call:
## lm(formula = price ~ poor_prop, data = house_prepared)
##
## Residuals:
##    Min     1Q Median     3Q    Max
##  -9.95  -4.00  -1.33   2.09  24.50
##
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  34.5820     0.5588    61.9   <2e-16 ***
## poor_prop    -0.9526     0.0385   -24.8   <2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## Residual standard error: 6.2 on 504 degrees of freedom
## Multiple R-squared:  0.549,	Adjusted R-squared:  0.548
## F-statistic:  613 on 1 and 504 DF,  p-value: <2e-16
```

In the model with `teachers` as an independent variable:

+ The slope of the regression line ($\beta_1$ estimate) tells us that the relation with `price` is positive (as we saw in the correlation plot). It means that when the proportion of teachers in the city rises, so does the house's price.
+ It's coefficient of determination $R^2$, lower than the other one, tells us that this model explains less of the variability embedded in the data.

In the model with `poor_prop` as the independent variable:

+ The negative sign of the coefficient indicates a negative relationship between those variables. It seems that, as the proportion of poor people in the city grows, the price of the household descent.

### Scatter plot of each model


```r
title <- "Scatter plot of the model 1"
house_prepared %>%
  ggplot(aes(teachers, price)) +
  geom_smooth(method = "lm", color = "black", fill = "grey39") +
  geom_point(shape = 1) +
  labs(title = title, x = "Teachers", y = "Price")
```

![]({{ base.url }}/images/blog/linear-regression/unnamed-chunk-19-1.png)<!-- -->

```r
title <- "Scatter plot of the model 2"
house_prepared %>%
  ggplot(aes(poor_prop, price)) +
  geom_smooth(method = "lm", color = "black", fill = "grey39") +
  geom_point(shape = 1) +
  labs(title = title, x = "Poor people proportion", y = "Price")
```

![]({{ base.url }}/images/blog/linear-regression/unnamed-chunk-19-2.png)<!-- -->


In the scatter plot of model 1:

+ We can see a trend between the data points, although the slope is mild.
+ The data points show up with a large dispersion around the regression line.
+ It stands out some vertical lines across the data points. It's quite clear near the `x` value of 20.
+ There's another horizontal pattern of this type at the `y` axis value of 50.

In the context of a Machine learning project, you shouldn't leave those details behind. It is something that I would check with the people responsible for the data collection process.

In the scatter plot of model 2:

+ You'll detect a more pronounced slope of the regression line.
+ Data points are closer to the regression line.
+ Once again, the pattern over `price = 50` shows up. It could mean that the data was capped for this value.

# Multiple Linear Regression (quantitative regressors)


```r
lm.multiple_quantitative <- lm(
  price ~ age + teachers + poor_prop, data = house_prepared
)
```

## Effect of each regressor and interpretation


```r
lm.multiple_quantitative %>% summary() %>% coefficients() %>% my_kable(full_width = T)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Estimate </th>
   <th style="text-align:right;"> Std. Error </th>
   <th style="text-align:right;"> t value </th>
   <th style="text-align:right;"> Pr(&gt;|t|) </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> (Intercept) </td>
   <td style="text-align:right;"> 6.64 </td>
   <td style="text-align:right;"> 3.00 </td>
   <td style="text-align:right;"> 2.2 </td>
   <td style="text-align:right;"> 0.03 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 0.04 </td>
   <td style="text-align:right;"> 0.01 </td>
   <td style="text-align:right;"> 3.5 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 1.15 </td>
   <td style="text-align:right;"> 0.13 </td>
   <td style="text-align:right;"> 9.1 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> -0.92 </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> -19.8 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
</tbody>
</table>

+ The parameter of `age` is very close to zero. It indicates that this variable has a small weight when determining the price of a house.
+ On `teachers` and `poor_prop`, you should note that they have more weighty parameters, although they are lower than the previous section's simple linear regression models.
+ The observed _t_ and the _p-value_ given for each descriptor variable determines the null hypothesis that the variable's coefficient is equal to zero.
+ For this model, all _p-values_ obtained are lower than the level of significance, and _t_ statistics are higher as the variable coefficient increases. It indicates that the variables are relevant to the model.

## Evaluation of the goodness of adjustment through the adjusted coefficient of determination

The adjusted coefficient of determination is another way to evaluate multiple linear regression models. You use it to soften the naturally occurring increase in the coefficient of determination $R^2$ as you add variables to the model.

You get it from the following expression.

$$
R^2_a = 1 - (1 - R^2)\frac{n-1}{n-p-1}
$$

Where,

+ $R^2_a$ is the adjusted $R^2$ or the adjusted coefficient of determination,
+ $R^2$ is the coefficient of determination,
+ $n$ is the number of observations in your data set, and
+ $p$ is the number of features in your model

You get it from the fitted model object like this.


```r
summary(lm.multiple_quantitative)$r.squared
```

```
## [1] 0.62
```

```r
summary(lm.multiple_quantitative)$adj.r.squared
```

```
## [1] 0.62
```

Concerning the standard $R^2$, the adjusted coefficient of determination is slightly lower, though not significantly. We'll get the most of it afterward.

## Extension of the model with the variables `room_num`, `n_hos_beds` and `n_hot_rooms`

Let's compare the previous model with the following one.


```r
lm.multiple_quantitative_extended <- lm(
  price ~ age + teachers + poor_prop + room_num + n_hos_beds + n_hot_rooms,
  data = house_prepared
)
```

So now, we see the statistical summary of the model.


```r
summary(lm.multiple_quantitative_extended)
```

```
##
## Call:
## lm(formula = price ~ age + teachers + poor_prop + room_num +
##     n_hos_beds + n_hot_rooms, data = house_prepared)
##
## Residuals:
##     Min      1Q  Median      3Q     Max
## -11.977  -3.044  -0.679   2.079  28.365
##
## Coefficients:
##              Estimate Std. Error t value Pr(>|t|)    
## (Intercept) -25.43960    4.69790   -5.42  9.5e-08 ***
## age           0.01841    0.01045    1.76   0.0786 .  
## teachers      0.97615    0.11610    8.41  4.4e-16 ***
## poor_prop    -0.61365    0.05111  -12.01  < 2e-16 ***
## room_num      4.81766    0.47401   10.16  < 2e-16 ***
## n_hos_beds    0.43317    0.15728    2.75   0.0061 **
## n_hot_rooms  -0.00204    0.01485   -0.14   0.8909    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## Residual standard error: 5.2 on 499 degrees of freedom
## Multiple R-squared:  0.688,	Adjusted R-squared:  0.685
## F-statistic:  184 on 6 and 499 DF,  p-value: <2e-16
```

+ The _t_ statistic and their associated p-values obtained for `age`, `n_hos_beds`, and `n_hot_rooms` tell us that these variables have a coefficient equal to zero in the model.

By introducing more explanatory variables than the previous model, it is of interest to use $R_a^2$ to compare them.


```r
summary(lm.multiple_quantitative)$adj.r.squared
```

```
## [1] 0.62
```

```r
summary(lm.multiple_quantitative_extended)$adj.r.squared
```

```
## [1] 0.68
```

In this case, we see an increase of more than 7% in the $R^2$ of the model.

# Multiple linear regression models (quantitative and qualitative regressors)

We want to know what happens if we extend the previous model with the `airport` variable.


```r
lm.multiple_quanti_quali <- lm(
  price ~
    age + teachers + poor_prop + room_num + n_hos_beds + n_hot_rooms + airport,
  data = house_prepared
)
```

Is the new model significantly better?


```r
summary(lm.multiple_quantitative)$adj.r.squared
```

```
## [1] 0.62
```

```r
summary(lm.multiple_quantitative_extended)$adj.r.squared
```

```
## [1] 0.68
```

In this case, the improvement obtained, comparing the adjusted $R^2$, is very small.

# Prediction of the price of housing

Now, let's imagine that we want to predict a new house's price with the following characteristics.

**_`age`=70, `teachers`=15, `poor_prop`=15, `room_num`=8, `n_hos_beds`=8, `n_hot_rooms`=100_**

**_[REVIEW: Is it the best one?]_**

I'll use the model fitted with quantitative variables to perform the prediction. So far, it's the best one.


```r
new_house <- tibble(
  age = 70,
  teachers = 15,
  poor_prop = 15,
  room_num = 8,
  n_hos_beds = 8,
  n_hot_rooms = 100
)

predict(
  lm.multiple_quantitative_extended,
  new_house,
  interval = "confidence"
) %>%
  data.frame() %>%
  my_kable(full_width = FALSE)
```

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> fit </th>
   <th style="text-align:right;"> lwr </th>
   <th style="text-align:right;"> upr </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 23 </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:right;"> 26 </td>
  </tr>
</tbody>
</table>

The model predicts a price in the range $(19,99, 25,20)$ with 95% confidence.

# Visual verification of modeling assumptions

I represent the residues against the values estimated by the model in a dispersion diagram.


```r
y_pred <- lm.multiple_quanti_quali$fitted.values
residuals <- summary(lm.multiple_quanti_quali)$residuals

residuals_df <- tibble(y_pred, residuals)

title <- "Residuals vs. Fitted"
residuals_df %>%
  ggplot(aes(y_pred, residuals)) +
  geom_smooth(color = "firebrick3", method = "lm", se = F) +
  geom_point(shape = 1) +
  labs(title = title, x = "Fitted Values", y = "Residuals")
```

```
## `geom_smooth()` using formula 'y ~ x'
```

![]({{ base.url }}/images/blog/linear-regression/unnamed-chunk-28-1.png)<!-- -->

The errors present homoscedasticity; that is, they are evenly distributed around the regression line for residuals, without forming any particular structure.

I check that the mean residue is zero and perform a histogram and hypothesis contrast to test the alternative hypothesis that the residues follow a normal distribution.


```r
residuals_df %>%
  ggplot(aes(residuals)) +
  geom_histogram(bins = 30, fill = "grey79", colour = "grey19") +
  labs(title = "Residuals distribution", x = "Residuals",  y = "Count")
```

![]({{ base.url }}/images/blog/linear-regression/unnamed-chunk-29-1.png)<!-- -->

They present a mean of almost 0, and I can accept the hypothesis that the residues follow a normal distribution according to the histogram and the p-value obtained in the normality test.

We can also catch outliers in the dependent variable where the residuals are too far from the line, using the z-score criterion.


```r
residuals_outliers <- residuals_df %>%
  rownames_to_column() %>%
  mutate(residuals_z = (residuals - mean(residuals)) / sd(residuals)) %>%
  filter(abs(residuals_z) > 3) %>%
  pull(rowname)

house_prepared %>% filter(row_number() %in% residuals_outliers) %>% my_kable() %>% scroll_box(width = "100%")
```

<div style="border: 1px solid #ddd; padding: 5px; overflow-x: scroll; width:100%; "><table class="table table-striped table-hover table-condensed table-responsive" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> price </th>
   <th style="text-align:right;"> resid_area </th>
   <th style="text-align:right;"> air_qual </th>
   <th style="text-align:right;"> room_num </th>
   <th style="text-align:right;"> age </th>
   <th style="text-align:right;"> dist1 </th>
   <th style="text-align:right;"> dist2 </th>
   <th style="text-align:right;"> dist3 </th>
   <th style="text-align:right;"> dist4 </th>
   <th style="text-align:right;"> teachers </th>
   <th style="text-align:right;"> poor_prop </th>
   <th style="text-align:left;"> airport </th>
   <th style="text-align:right;"> n_hos_beds </th>
   <th style="text-align:right;"> n_hot_rooms </th>
   <th style="text-align:left;"> waterbody </th>
   <th style="text-align:right;"> rainfall </th>
   <th style="text-align:left;"> bus_ter </th>
   <th style="text-align:right;"> parks </th>
   <th style="text-align:right;"> Sold </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 50 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.63 </td>
   <td style="text-align:right;"> 5.0 </td>
   <td style="text-align:right;"> 100.0 </td>
   <td style="text-align:right;"> 1.5 </td>
   <td style="text-align:right;"> 1.11 </td>
   <td style="text-align:right;"> 1.5 </td>
   <td style="text-align:right;"> 1.23 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 3.3 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.7 </td>
   <td style="text-align:right;"> 117 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 41 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.60 </td>
   <td style="text-align:right;"> 4.6 </td>
   <td style="text-align:right;"> 100.0 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 1.54 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 1.13 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 34.4 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 8.4 </td>
   <td style="text-align:right;"> 151 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50 </td>
   <td style="text-align:right;"> 36 </td>
   <td style="text-align:right;"> 0.50 </td>
   <td style="text-align:right;"> 6.2 </td>
   <td style="text-align:right;"> 83.0 </td>
   <td style="text-align:right;"> 3.1 </td>
   <td style="text-align:right;"> 2.61 </td>
   <td style="text-align:right;"> 3.0 </td>
   <td style="text-align:right;"> 2.88 </td>
   <td style="text-align:right;"> 23 </td>
   <td style="text-align:right;"> 4.6 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 7.5 </td>
   <td style="text-align:right;"> 117 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50 </td>
   <td style="text-align:right;"> 34 </td>
   <td style="text-align:right;"> 0.65 </td>
   <td style="text-align:right;"> 6.2 </td>
   <td style="text-align:right;"> 86.9 </td>
   <td style="text-align:right;"> 2.1 </td>
   <td style="text-align:right;"> 1.53 </td>
   <td style="text-align:right;"> 1.8 </td>
   <td style="text-align:right;"> 1.76 </td>
   <td style="text-align:right;"> 27 </td>
   <td style="text-align:right;"> 5.1 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 8.6 </td>
   <td style="text-align:right;"> 117 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 54 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.63 </td>
   <td style="text-align:right;"> 6.7 </td>
   <td style="text-align:right;"> 96.8 </td>
   <td style="text-align:right;"> 1.6 </td>
   <td style="text-align:right;"> 1.28 </td>
   <td style="text-align:right;"> 1.6 </td>
   <td style="text-align:right;"> 0.94 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 3.7 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 6.7 </td>
   <td style="text-align:right;"> 117 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 58 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.07 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.63 </td>
   <td style="text-align:right;"> 7.0 </td>
   <td style="text-align:right;"> 97.5 </td>
   <td style="text-align:right;"> 1.4 </td>
   <td style="text-align:right;"> 0.92 </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 1.28 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 3.0 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 10.1 </td>
   <td style="text-align:right;"> 117 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 46 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.63 </td>
   <td style="text-align:right;"> 6.2 </td>
   <td style="text-align:right;"> 100.0 </td>
   <td style="text-align:right;"> 1.4 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 1.4 </td>
   <td style="text-align:right;"> 0.99 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 9.5 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.8 </td>
   <td style="text-align:right;"> 117 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 25 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50 </td>
   <td style="text-align:right;"> 48 </td>
   <td style="text-align:right;"> 0.67 </td>
   <td style="text-align:right;"> 5.9 </td>
   <td style="text-align:right;"> 89.6 </td>
   <td style="text-align:right;"> 1.1 </td>
   <td style="text-align:right;"> 1.01 </td>
   <td style="text-align:right;"> 1.2 </td>
   <td style="text-align:right;"> 1.12 </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 8.9 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 10.8 </td>
   <td style="text-align:right;"> 117 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 57 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 24 </td>
   <td style="text-align:right;"> 41 </td>
   <td style="text-align:right;"> 0.49 </td>
   <td style="text-align:right;"> 5.4 </td>
   <td style="text-align:right;"> 9.8 </td>
   <td style="text-align:right;"> 3.7 </td>
   <td style="text-align:right;"> 3.48 </td>
   <td style="text-align:right;"> 3.7 </td>
   <td style="text-align:right;"> 3.53 </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:right;"> 29.6 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 5.7 </td>
   <td style="text-align:right;"> 112 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 49 </td>
   <td style="text-align:right;"> 34 </td>
   <td style="text-align:right;"> 0.65 </td>
   <td style="text-align:right;"> 6.2 </td>
   <td style="text-align:right;"> 91.5 </td>
   <td style="text-align:right;"> 2.5 </td>
   <td style="text-align:right;"> 2.04 </td>
   <td style="text-align:right;"> 2.4 </td>
   <td style="text-align:right;"> 2.17 </td>
   <td style="text-align:right;"> 27 </td>
   <td style="text-align:right;"> 5.9 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 10.1 </td>
   <td style="text-align:right;"> 154 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 24 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
</tbody>
</table></div>

We see that practically all cases with `price = 50` appear as atypical cases. Other outliers come out among the variables where we couldn't establish a clear boundary for atypical cases, e.g., for `teachers > 27 and poor_prop > 29` (I mean [here](missing-values)).

It will be interesting to see what happens if we adjust the model again, leaving out the outliers detected by analyzing residuals. How do you think that this will affect the model's performance?


```r
lm(
  price ~ age + teachers + poor_prop + room_num + n_hos_beds + n_hot_rooms,
  data = house_prepared %>% filter(!row_number() %in% residuals_outliers)
) %>%
  summary() %>% .$r.squared
```

```
## [1] 0.78
```

The $R^2$ of the model improves by almost 10 points regarding the model that includes the outliers!

# Takeaways

This is what we have learned in this blog post:

+ Linear Regression is relatively simple, and so, it is more explainable than other models.
+ Be aware of the outliers. Give the data processing step the care that it deserves. Recheck them when you go through the validation of the model's assumptions.
+ Check the model assumptions (independent variable or residuals normally distributed and homoscedasticity) to ensure that what you are doing is right.
+ Test different variable combinations to see how you can improve your model's performance.
