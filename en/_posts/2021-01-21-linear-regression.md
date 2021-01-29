---
title: "Linear Regression with R"
author: "Carlos Dávila Luelmo"
date: "2021-01-21"
tags: [R, Linear Regression, Statistics, Inference, Housing]
---



In this blog entry, I'd like to showcase a Linear Regression implementation. I'll do it with a data set about housing and `R`. Moreover, I assume that you readers have a base knowledge about statistical inference.

The aim here is to use a Linear Regression model, concentrating more on inference than prediction (although we can perform it anyway)—-this way, we focus more on the relationships between the variables rather than the model's prediction power.

A Linear Regression is a math model used to approach the dependence relationships between a dependent variable $Y$, a set of independent variables $X_i$, and a random term $\varepsilon$. Its equation has this form:

$$
Y_t = \beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + ... + + \beta_p \cdot X_p + \varepsilon
$$

Simply put, the linear regression attempts to **generalize the relation between two or more variables** by minimizing the distance between every data point and the line. That is why we say that it uses the least-squares method to fit the model.

I will not get into many details, as it is a widely known concept. You can find multiple excellent explanations and resources out there if you do a quick search for the term at the search engine of your preference.

Another thing to point out about linear regression is that it is a relatively simple model. We can understand and so explain what is doing under the hood when we analyze it. It is one of the most used Machine Learning algorithms for that reason.

The variable that we want to model here is the house's price, as you will see afterward.

The aim is to go through the modeling process. It will be more enjoyable for you if I show a hands-on implementation of linear regression.

## The level of significance and the performance metric

Before We go into the action, I'll spot some assumptions to have them in mind throughout the exercise.

The **significance's level** established for our purpose will be an $\alpha = 0.05$. It is an often confusing term to explain. For me, the best way to put it is this.

In statistics, we are often testing a hypothesis. However, we are pessimistic guys, so we usually try the opposite of the hypothesis we want to prove. For example, when we want to check if a variable is normally distributed, we test if it is not normally distributed as our initial hypothesis.

The significance level is the probability that we have to get the result if our initial hypothesis (or null hypothesis) is correct. That means if we have a chance to find this result lower than our $\alpha = 0.05$ --a very low probability--we reject the null hypothesis and vice versa.

I choose this $\alpha$ because it's a convention, but we may have selected another value. The value we chose gives us different type I and II error ratios. Check [this excellent explanation about type I and II errors to learn more](https://www.youtube.com/watch?v=edzQQFNzFjM).

**The performance metric** most used in a Linear Regression is the determination coefficient $R^2$--or its cousin, the adjusted $R^2$. In short, it measures the proportion of variability contained in our data explained by the corresponding model. It ranges between 0 and 1.

Think about the model as a way of generalizing. You do this all the time. For instance, imagine that you have a model in your mind regarding roses that generalize your knowledge about roses following the expression: "All roses are red or white". Your model cannot explain the variability that you will find if you, let's say, do a quick search about roses on DuckDuckGo.

With this established, we can commence.

## The working environment

First of all, set the working libraries that you will need. Remember to keep it tidy. The fewer dependencies, the more durable and reproducible the work will be.

I will use my favorite `R` libraries like the `tidyverse` package, but this is only a possible solution.


```r
library(tidyverse)
library(GGally)
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



## The Dataset

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

Once you've digested the context, the next step is to take a glimpse at our data structure.

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
   <td style="text-align:right;"> 5.0000 </td>
   <td style="text-align:right;"> 17.0250 </td>
   <td style="text-align:right;"> 21.2000 </td>
   <td style="text-align:right;"> 22.5289 </td>
   <td style="text-align:right;"> 25.0000 </td>
   <td style="text-align:right;"> 50.0000 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resid_area </td>
   <td style="text-align:right;"> 30.4600 </td>
   <td style="text-align:right;"> 35.1900 </td>
   <td style="text-align:right;"> 39.6900 </td>
   <td style="text-align:right;"> 41.1368 </td>
   <td style="text-align:right;"> 48.1000 </td>
   <td style="text-align:right;"> 57.7400 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> air_qual </td>
   <td style="text-align:right;"> 0.3850 </td>
   <td style="text-align:right;"> 0.4490 </td>
   <td style="text-align:right;"> 0.5380 </td>
   <td style="text-align:right;"> 0.5547 </td>
   <td style="text-align:right;"> 0.6240 </td>
   <td style="text-align:right;"> 0.8710 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> room_num </td>
   <td style="text-align:right;"> 3.5610 </td>
   <td style="text-align:right;"> 5.8855 </td>
   <td style="text-align:right;"> 6.2085 </td>
   <td style="text-align:right;"> 6.2846 </td>
   <td style="text-align:right;"> 6.6235 </td>
   <td style="text-align:right;"> 8.7800 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 2.9000 </td>
   <td style="text-align:right;"> 45.0250 </td>
   <td style="text-align:right;"> 77.5000 </td>
   <td style="text-align:right;"> 68.5749 </td>
   <td style="text-align:right;"> 94.0750 </td>
   <td style="text-align:right;"> 100.0000 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist1 </td>
   <td style="text-align:right;"> 1.1300 </td>
   <td style="text-align:right;"> 2.2700 </td>
   <td style="text-align:right;"> 3.3850 </td>
   <td style="text-align:right;"> 3.9720 </td>
   <td style="text-align:right;"> 5.3675 </td>
   <td style="text-align:right;"> 12.3200 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist2 </td>
   <td style="text-align:right;"> 0.9200 </td>
   <td style="text-align:right;"> 1.9400 </td>
   <td style="text-align:right;"> 3.0100 </td>
   <td style="text-align:right;"> 3.6288 </td>
   <td style="text-align:right;"> 4.9925 </td>
   <td style="text-align:right;"> 11.9300 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist3 </td>
   <td style="text-align:right;"> 1.1500 </td>
   <td style="text-align:right;"> 2.2325 </td>
   <td style="text-align:right;"> 3.3750 </td>
   <td style="text-align:right;"> 3.9607 </td>
   <td style="text-align:right;"> 5.4075 </td>
   <td style="text-align:right;"> 12.3200 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist4 </td>
   <td style="text-align:right;"> 0.7300 </td>
   <td style="text-align:right;"> 1.9400 </td>
   <td style="text-align:right;"> 3.0700 </td>
   <td style="text-align:right;"> 3.6190 </td>
   <td style="text-align:right;"> 4.9850 </td>
   <td style="text-align:right;"> 11.9400 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 18.0000 </td>
   <td style="text-align:right;"> 19.8000 </td>
   <td style="text-align:right;"> 20.9500 </td>
   <td style="text-align:right;"> 21.5445 </td>
   <td style="text-align:right;"> 22.6000 </td>
   <td style="text-align:right;"> 27.4000 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> 1.7300 </td>
   <td style="text-align:right;"> 6.9500 </td>
   <td style="text-align:right;"> 11.3600 </td>
   <td style="text-align:right;"> 12.6531 </td>
   <td style="text-align:right;"> 16.9550 </td>
   <td style="text-align:right;"> 37.9700 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hos_beds </td>
   <td style="text-align:right;"> 5.2680 </td>
   <td style="text-align:right;"> 6.6345 </td>
   <td style="text-align:right;"> 7.9990 </td>
   <td style="text-align:right;"> 7.8998 </td>
   <td style="text-align:right;"> 9.0880 </td>
   <td style="text-align:right;"> 10.8760 </td>
   <td style="text-align:right;"> 8 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hot_rooms </td>
   <td style="text-align:right;"> 10.0600 </td>
   <td style="text-align:right;"> 100.5940 </td>
   <td style="text-align:right;"> 117.2840 </td>
   <td style="text-align:right;"> 98.8172 </td>
   <td style="text-align:right;"> 141.1000 </td>
   <td style="text-align:right;"> 153.9040 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> rainfall </td>
   <td style="text-align:right;"> 3.0000 </td>
   <td style="text-align:right;"> 28.0000 </td>
   <td style="text-align:right;"> 39.0000 </td>
   <td style="text-align:right;"> 39.1818 </td>
   <td style="text-align:right;"> 50.0000 </td>
   <td style="text-align:right;"> 60.0000 </td>
   <td style="text-align:right;">  </td>
  </tr>
  <tr>
   <td style="text-align:left;"> parks </td>
   <td style="text-align:right;"> 0.0333 </td>
   <td style="text-align:right;"> 0.0465 </td>
   <td style="text-align:right;"> 0.0535 </td>
   <td style="text-align:right;"> 0.0545 </td>
   <td style="text-align:right;"> 0.0614 </td>
   <td style="text-align:right;"> 0.0867 </td>
   <td style="text-align:right;">  </td>
  </tr>
</tbody>
</table>

+ See the eight missing values in `n_hos_beds`? We need to handle these values.
+ All the rest of the features are complete.

Let's do some plots!

First, a scatter matrix that will give us much relevant information. It has:

+ the response variable of our model, `price`, plotted against the rest of the variables;
+ frequency diagrams at the diagonal, so you get a glance of the data distribution;
+ scatter plots at the bottom to see how variables are related to each other, and
+ Pearson's correlation coefficients at the upper section. [Check this Wikipedia article](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) if you are not familiar with it.


```r
my_ggpairs <- function(data, num_vars){
  num_vars <- enquo(num_vars)

  data %>%
    filter(across(everything(), ~!is.na(.x))) %>%
    select(!!num_vars) %>%
    ggpairs(lower = list(continuous = wrap("points", color = "#333333", shape = 1, size = .5))) +
    theme(axis.text = element_text(size = 6),
          panel.grid.major.x = element_blank(),
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

+ Variables like `room_num`, `teachers`, and `poor_prop`, are linearly related to the `price` feature. It can be glimpsed in the scatter plots, and they present a correlation coefficient between -1 and -0.5 and between 0.5 and 1, meaning that those relationships are considered strong (it is a convention).
+ All `dist` variables are highly correlated with each other. This high level of correlation between explanatory variables is known as _multicollinearity_. We are interested in linear relationships between predictors and the response variable. Nevertheless, when predictors are correlated to each other, they do not give us relevant information. Commonly, you only take one of the variables that present _multicolinearity_ and leave the others out of the model.
+ `n_hot_rooms` takes a range of discrete values, although it is a numeric variable. It has to do with the data collection process or could be a posterior aggregation. By now, let's consider it as numeric.
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
   <td style="text-align:right;"> 9.1822 </td>
   <td style="text-align:right;"> 84.3124 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resid_area </td>
   <td style="text-align:right;"> 6.8604 </td>
   <td style="text-align:right;"> 47.0644 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> air_qual </td>
   <td style="text-align:right;"> 0.1159 </td>
   <td style="text-align:right;"> 0.0134 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> room_num </td>
   <td style="text-align:right;"> 0.7026 </td>
   <td style="text-align:right;"> 0.4937 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 28.1489 </td>
   <td style="text-align:right;"> 792.3584 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist1 </td>
   <td style="text-align:right;"> 2.1085 </td>
   <td style="text-align:right;"> 4.4459 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist2 </td>
   <td style="text-align:right;"> 2.1086 </td>
   <td style="text-align:right;"> 4.4461 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist3 </td>
   <td style="text-align:right;"> 2.1198 </td>
   <td style="text-align:right;"> 4.4935 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist4 </td>
   <td style="text-align:right;"> 2.0992 </td>
   <td style="text-align:right;"> 4.4067 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 2.1649 </td>
   <td style="text-align:right;"> 4.6870 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> 7.1411 </td>
   <td style="text-align:right;"> 50.9948 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hos_beds </td>
   <td style="text-align:right;"> 1.4767 </td>
   <td style="text-align:right;"> 2.1806 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hot_rooms </td>
   <td style="text-align:right;"> 51.5786 </td>
   <td style="text-align:right;"> 2660.3489 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> rainfall </td>
   <td style="text-align:right;"> 12.5137 </td>
   <td style="text-align:right;"> 156.5926 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> parks </td>
   <td style="text-align:right;"> 0.0106 </td>
   <td style="text-align:right;"> 0.0001 </td>
  </tr>
</tbody>
</table>

We can now see that `air_qual` has a very low standard deviation and variance, and `parks` shows near-zero variance.

Both quantitative features ranges are very narrow, as we saw at the beginning of this section. In advance, we'd think that those near-zero variances mean that the variables do not hold decisive information. However, we have to be sure before removing any information at all of the data set.

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
   <td style="text-align:right;"> 2.000 </td>
   <td style="text-align:right;"> 45.0593 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> resid_area </td>
   <td style="text-align:right;"> 4.400 </td>
   <td style="text-align:right;"> 15.0198 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> air_qual </td>
   <td style="text-align:right;"> 1.278 </td>
   <td style="text-align:right;"> 16.0079 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> room_num </td>
   <td style="text-align:right;"> 1.000 </td>
   <td style="text-align:right;"> 88.1423 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 10.750 </td>
   <td style="text-align:right;"> 70.3557 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist1 </td>
   <td style="text-align:right;"> 1.000 </td>
   <td style="text-align:right;"> 66.9960 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist2 </td>
   <td style="text-align:right;"> 1.200 </td>
   <td style="text-align:right;"> 69.9605 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist3 </td>
   <td style="text-align:right;"> 1.000 </td>
   <td style="text-align:right;"> 66.9960 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> dist4 </td>
   <td style="text-align:right;"> 1.400 </td>
   <td style="text-align:right;"> 69.7628 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 4.118 </td>
   <td style="text-align:right;"> 9.0909 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> 1.000 </td>
   <td style="text-align:right;"> 89.9209 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> airport </td>
   <td style="text-align:right;"> 1.200 </td>
   <td style="text-align:right;"> 0.3953 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hos_beds </td>
   <td style="text-align:right;"> 1.000 </td>
   <td style="text-align:right;"> 89.7233 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> n_hot_rooms </td>
   <td style="text-align:right;"> 1.000 </td>
   <td style="text-align:right;"> 76.4822 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> waterbody </td>
   <td style="text-align:right;"> 1.181 </td>
   <td style="text-align:right;"> 0.7905 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> rainfall </td>
   <td style="text-align:right;"> 1.053 </td>
   <td style="text-align:right;"> 8.3004 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> bus_ter </td>
   <td style="text-align:right;"> 0.000 </td>
   <td style="text-align:right;"> 0.1976 </td>
   <td style="text-align:left;"> TRUE </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> parks </td>
   <td style="text-align:right;"> 1.000 </td>
   <td style="text-align:right;"> 94.4664 </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> FALSE </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Sold </td>
   <td style="text-align:right;"> 1.200 </td>
   <td style="text-align:right;"> 0.3953 </td>
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
   <td style="text-align:right;"> 39.69 </td>
   <td style="text-align:right;"> 0.585 </td>
   <td style="text-align:right;"> 5.390 </td>
   <td style="text-align:right;"> 72.9 </td>
   <td style="text-align:right;"> 2.86 </td>
   <td style="text-align:right;"> 2.61 </td>
   <td style="text-align:right;"> 2.98 </td>
   <td style="text-align:right;"> 2.74 </td>
   <td style="text-align:right;"> 20.8 </td>
   <td style="text-align:right;"> 21.14 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 121.58 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 44 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0610 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 22.6 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.770 </td>
   <td style="text-align:right;"> 6.112 </td>
   <td style="text-align:right;"> 81.3 </td>
   <td style="text-align:right;"> 2.78 </td>
   <td style="text-align:right;"> 2.38 </td>
   <td style="text-align:right;"> 2.56 </td>
   <td style="text-align:right;"> 2.31 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 12.67 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 141.81 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 26 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0742 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 25.0 </td>
   <td style="text-align:right;"> 40.59 </td>
   <td style="text-align:right;"> 0.489 </td>
   <td style="text-align:right;"> 6.182 </td>
   <td style="text-align:right;"> 42.4 </td>
   <td style="text-align:right;"> 4.15 </td>
   <td style="text-align:right;"> 3.81 </td>
   <td style="text-align:right;"> 3.96 </td>
   <td style="text-align:right;"> 3.87 </td>
   <td style="text-align:right;"> 21.4 </td>
   <td style="text-align:right;"> 9.47 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 12.20 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 30 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0479 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 18.8 </td>
   <td style="text-align:right;"> 40.01 </td>
   <td style="text-align:right;"> 0.547 </td>
   <td style="text-align:right;"> 5.913 </td>
   <td style="text-align:right;"> 92.9 </td>
   <td style="text-align:right;"> 2.55 </td>
   <td style="text-align:right;"> 2.23 </td>
   <td style="text-align:right;"> 2.56 </td>
   <td style="text-align:right;"> 2.07 </td>
   <td style="text-align:right;"> 22.2 </td>
   <td style="text-align:right;"> 16.21 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 151.50 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 35 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0576 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 19.7 </td>
   <td style="text-align:right;"> 35.64 </td>
   <td style="text-align:right;"> 0.439 </td>
   <td style="text-align:right;"> 5.963 </td>
   <td style="text-align:right;"> 45.7 </td>
   <td style="text-align:right;"> 7.08 </td>
   <td style="text-align:right;"> 6.55 </td>
   <td style="text-align:right;"> 7.00 </td>
   <td style="text-align:right;"> 6.63 </td>
   <td style="text-align:right;"> 23.2 </td>
   <td style="text-align:right;"> 13.45 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 111.58 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0404 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 33.8 </td>
   <td style="text-align:right;"> 33.97 </td>
   <td style="text-align:right;"> 0.647 </td>
   <td style="text-align:right;"> 7.203 </td>
   <td style="text-align:right;"> 81.8 </td>
   <td style="text-align:right;"> 2.12 </td>
   <td style="text-align:right;"> 1.95 </td>
   <td style="text-align:right;"> 2.37 </td>
   <td style="text-align:right;"> 2.01 </td>
   <td style="text-align:right;"> 27.0 </td>
   <td style="text-align:right;"> 9.59 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 112.70 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0680 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 7.5 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.679 </td>
   <td style="text-align:right;"> 6.782 </td>
   <td style="text-align:right;"> 90.8 </td>
   <td style="text-align:right;"> 1.90 </td>
   <td style="text-align:right;"> 1.54 </td>
   <td style="text-align:right;"> 2.04 </td>
   <td style="text-align:right;"> 1.80 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 25.79 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 10.06 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 35 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0646 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.3 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.693 </td>
   <td style="text-align:right;"> 5.349 </td>
   <td style="text-align:right;"> 96.0 </td>
   <td style="text-align:right;"> 1.75 </td>
   <td style="text-align:right;"> 1.38 </td>
   <td style="text-align:right;"> 1.88 </td>
   <td style="text-align:right;"> 1.80 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 19.77 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;">  </td>
   <td style="text-align:right;"> 150.66 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0677 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table></div>

No other value in the rest of the features raises suspicion.

Let's replace the missing values in the `n_hos_beds` with the median of the variable. It is just one valid approach, which I consider preferable because you do not lose information. Besides, there are only a few missing values.


```r
house_complete <- house %>%
  mutate(n_hos_beds = case_when(
    is.na(n_hos_beds) ~ median(n_hos_beds, na.rm = TRUE),
    TRUE ~ n_hos_beds
  ))
```

Outliers are a pain in the neck when you want to fit a particular type of model. Linear Regression is one of them.

An excellent tool to detect outliers are box plots.


```r
house_complete %>%
  select(all_of(num_vars)) %>%
  pivot_longer(names(.), names_to = "variable", values_to = "valor") %>% ggplot(aes(variable, valor)) +
  geom_boxplot(fill = "grey89", outlier.shape = 1) +
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
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.693 </td>
   <td style="text-align:right;"> 5.453 </td>
   <td style="text-align:right;"> 100.0 </td>
   <td style="text-align:right;"> 1.57 </td>
   <td style="text-align:right;"> 1.26 </td>
   <td style="text-align:right;"> 1.79 </td>
   <td style="text-align:right;"> 1.34 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 30.59 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.30 </td>
   <td style="text-align:right;"> 13.04 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 26 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0653 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 12 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.614 </td>
   <td style="text-align:right;"> 5.304 </td>
   <td style="text-align:right;"> 97.3 </td>
   <td style="text-align:right;"> 2.28 </td>
   <td style="text-align:right;"> 1.99 </td>
   <td style="text-align:right;"> 2.41 </td>
   <td style="text-align:right;"> 1.73 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 24.91 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.34 </td>
   <td style="text-align:right;"> 15.10 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 39 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0619 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 51.89 </td>
   <td style="text-align:right;"> 0.624 </td>
   <td style="text-align:right;"> 6.174 </td>
   <td style="text-align:right;"> 93.6 </td>
   <td style="text-align:right;"> 1.86 </td>
   <td style="text-align:right;"> 1.54 </td>
   <td style="text-align:right;"> 1.87 </td>
   <td style="text-align:right;"> 1.18 </td>
   <td style="text-align:right;"> 18.8 </td>
   <td style="text-align:right;"> 24.16 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 5.68 </td>
   <td style="text-align:right;"> 10.11 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 28 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0570 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 18 </td>
   <td style="text-align:right;"> 51.89 </td>
   <td style="text-align:right;"> 0.624 </td>
   <td style="text-align:right;"> 6.431 </td>
   <td style="text-align:right;"> 98.8 </td>
   <td style="text-align:right;"> 1.96 </td>
   <td style="text-align:right;"> 1.61 </td>
   <td style="text-align:right;"> 1.92 </td>
   <td style="text-align:right;"> 1.77 </td>
   <td style="text-align:right;"> 18.8 </td>
   <td style="text-align:right;"> 15.39 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 8.16 </td>
   <td style="text-align:right;"> 14.14 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 41 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0564 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 19 </td>
   <td style="text-align:right;"> 35.19 </td>
   <td style="text-align:right;"> 0.515 </td>
   <td style="text-align:right;"> 5.985 </td>
   <td style="text-align:right;"> 45.4 </td>
   <td style="text-align:right;"> 4.89 </td>
   <td style="text-align:right;"> 4.64 </td>
   <td style="text-align:right;"> 5.05 </td>
   <td style="text-align:right;"> 4.67 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 9.74 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 6.38 </td>
   <td style="text-align:right;"> 11.15 </td>
   <td style="text-align:left;"> Lake </td>
   <td style="text-align:right;"> 28 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0477 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:right;"> 35.96 </td>
   <td style="text-align:right;"> 0.499 </td>
   <td style="text-align:right;"> 5.841 </td>
   <td style="text-align:right;"> 61.4 </td>
   <td style="text-align:right;"> 3.39 </td>
   <td style="text-align:right;"> 3.28 </td>
   <td style="text-align:right;"> 3.62 </td>
   <td style="text-align:right;"> 3.22 </td>
   <td style="text-align:right;"> 20.8 </td>
   <td style="text-align:right;"> 11.41 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 7.50 </td>
   <td style="text-align:right;"> 15.16 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 39 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0454 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
</tbody>
</table></div>

The rest of the features show values within the inner quantiles.

There are two things that we can do. Drop those observations or replace the outlier values with other (e.g., the mean, a.k.a. the **expected** value or the minimum).

In another context, we may have the chance to get more information about the data set we are dealing with, but we cannot go any further in this case.

I'll consider these outliers as errors in the data collection process and replace them with a central value like the median.

<span style="color: grey;">In the next code snippet I save my cleaned data in the new `house_prepared` variable.</span>


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
+ Leaving out those values observed at more than three standard deviations of the mean (absolute value of z greater than 3), which will be zero after the standardization.

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
   <td style="text-align:right;"> 3.561 </td>
   <td style="text-align:right;"> -3.876 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3.863 </td>
   <td style="text-align:right;"> -3.447 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4.138 </td>
   <td style="text-align:right;"> -3.055 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4.138 </td>
   <td style="text-align:right;"> -3.055 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.398 </td>
   <td style="text-align:right;"> 3.008 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.704 </td>
   <td style="text-align:right;"> 3.443 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.725 </td>
   <td style="text-align:right;"> 3.473 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 8.780 </td>
   <td style="text-align:right;"> 3.551 </td>
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

In `R`, we use the formula syntax. It's a very intuitive way of writing your model. You place your target variable on the left-hand side of the formula and the features you want on the right-hand side and split them with a tilde symbol (`~`).

<center>
`target ~ features`
</center>

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
   <td style="text-align:right;"> -23.676 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 2.145 </td>
  </tr>
</tbody>
</table>

So, we can represent the model with the following expression.

$$\overline{y} = -23.676 +2.145\cdot teachers$$

And the coefficients for the second one.


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
   <td style="text-align:right;"> 34.5820 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> -0.9526 </td>
  </tr>
</tbody>
</table>

Moreover, the model would be the next.

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
## Residual standard error: 7.93 on 504 degrees of freedom
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
## Residual standard error: 6.17 on 504 degrees of freedom
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
  geom_smooth(method = "lm", color = "firebrick3", fill = NA) +
  geom_point(shape = 1) +
  labs(title = title, x = "Teachers", y = "Price")
```

![]({{ base.url }}/images/blog/linear-regression/unnamed-chunk-19-1.png)<!-- -->

```r
title <- "Scatter plot of the model 2"
house_prepared %>%
  ggplot(aes(poor_prop, price)) +
  geom_smooth(method = "lm", color = "firebrick3", fill = NA) +
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
   <td style="text-align:right;"> 6.6416 </td>
   <td style="text-align:right;"> 2.9960 </td>
   <td style="text-align:right;"> 2.217 </td>
   <td style="text-align:right;"> 0.0271 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> age </td>
   <td style="text-align:right;"> 0.0400 </td>
   <td style="text-align:right;"> 0.0113 </td>
   <td style="text-align:right;"> 3.545 </td>
   <td style="text-align:right;"> 0.0004 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> teachers </td>
   <td style="text-align:right;"> 1.1489 </td>
   <td style="text-align:right;"> 0.1261 </td>
   <td style="text-align:right;"> 9.109 </td>
   <td style="text-align:right;"> 0.0000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> poor_prop </td>
   <td style="text-align:right;"> -0.9172 </td>
   <td style="text-align:right;"> 0.0462 </td>
   <td style="text-align:right;"> -19.837 </td>
   <td style="text-align:right;"> 0.0000 </td>
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
## [1] 0.6191
```

```r
summary(lm.multiple_quantitative)$adj.r.squared
```

```
## [1] 0.6168
```

Concerning the standard $R^2$, the adjusted coefficient of determination is practically the same. We'll get the most of this metric afterward.

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
## Residual standard error: 5.16 on 499 degrees of freedom
## Multiple R-squared:  0.688,	Adjusted R-squared:  0.685
## F-statistic:  184 on 6 and 499 DF,  p-value: <2e-16
```

+ The _t_ statistic and their associated p-values obtained for `age` and `n_hot_rooms` tell us that these variables are not significant for the model.

However, one thing is that variables are not significant, and the other is that the model is not significant.

In the same inference approach of the modeling process, we use the _F-statistic_ to check if a model significantly explains the response variable `Y` or not. If we get an _F_ close to one, it means that the model is not significant. But here, we got an _F_ way higher than one and a small _p_. So, we can consider that our variables are significantly linearly correlated with our response variable.

As we introduced more explanatory variables than the previous model, it is of interest to use $R_a^2$ to compare them.


```r
summary(lm.multiple_quantitative)$adj.r.squared
```

```
## [1] 0.6168
```

```r
summary(lm.multiple_quantitative_extended)$adj.r.squared
```

```
## [1] 0.6845
```

In this case, we see an increase of about 6% in the $R^2$ of the model.

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
summary(lm.multiple_quantitative_extended)$adj.r.squared
```

```
## [1] 0.6845
```

```r
summary(lm.multiple_quanti_quali)$adj.r.squared
```

```
## [1] 0.6882
```

In this case, the improvement obtained, comparing the adjusted $R^2$, is very small.

# Prediction of the price of housing

Now, let's imagine that we want to predict a new house's price with the following characteristics.

**`age = 70`, `teachers = 15`, `poor_prop = 15`, `room_num = 8`, `n_hos_beds = 8`, `n_hot_rooms = 100`**

I'll use the model fitted with quantitative and qualitative variables to perform the prediction. So far, it's the best one.


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
   <td style="text-align:right;"> 23.09 </td>
   <td style="text-align:right;"> 20.52 </td>
   <td style="text-align:right;"> 25.66 </td>
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
  geom_point(shape = 1, color = "#333333") +
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
  geom_histogram(bins = 30, fill = "#333333", colour = "white") +
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
   <td style="text-align:right;"> 50.0 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.631 </td>
   <td style="text-align:right;"> 4.970 </td>
   <td style="text-align:right;"> 100.0 </td>
   <td style="text-align:right;"> 1.47 </td>
   <td style="text-align:right;"> 1.11 </td>
   <td style="text-align:right;"> 1.52 </td>
   <td style="text-align:right;"> 1.23 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 3.26 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.700 </td>
   <td style="text-align:right;"> 117.3 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 41 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0624 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 17.9 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.597 </td>
   <td style="text-align:right;"> 4.628 </td>
   <td style="text-align:right;"> 100.0 </td>
   <td style="text-align:right;"> 1.77 </td>
   <td style="text-align:right;"> 1.54 </td>
   <td style="text-align:right;"> 1.78 </td>
   <td style="text-align:right;"> 1.13 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 34.37 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 8.358 </td>
   <td style="text-align:right;"> 151.4 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0587 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50.0 </td>
   <td style="text-align:right;"> 36.20 </td>
   <td style="text-align:right;"> 0.504 </td>
   <td style="text-align:right;"> 6.208 </td>
   <td style="text-align:right;"> 83.0 </td>
   <td style="text-align:right;"> 3.13 </td>
   <td style="text-align:right;"> 2.61 </td>
   <td style="text-align:right;"> 2.95 </td>
   <td style="text-align:right;"> 2.88 </td>
   <td style="text-align:right;"> 22.6 </td>
   <td style="text-align:right;"> 4.63 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 7.500 </td>
   <td style="text-align:right;"> 117.3 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 20 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0570 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50.0 </td>
   <td style="text-align:right;"> 33.97 </td>
   <td style="text-align:right;"> 0.647 </td>
   <td style="text-align:right;"> 6.208 </td>
   <td style="text-align:right;"> 86.9 </td>
   <td style="text-align:right;"> 2.09 </td>
   <td style="text-align:right;"> 1.53 </td>
   <td style="text-align:right;"> 1.83 </td>
   <td style="text-align:right;"> 1.76 </td>
   <td style="text-align:right;"> 27.0 </td>
   <td style="text-align:right;"> 5.12 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 8.600 </td>
   <td style="text-align:right;"> 117.3 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 54 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0552 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50.0 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.631 </td>
   <td style="text-align:right;"> 6.683 </td>
   <td style="text-align:right;"> 96.8 </td>
   <td style="text-align:right;"> 1.55 </td>
   <td style="text-align:right;"> 1.28 </td>
   <td style="text-align:right;"> 1.65 </td>
   <td style="text-align:right;"> 0.94 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 3.73 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 6.700 </td>
   <td style="text-align:right;"> 117.3 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 58 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0675 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50.0 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.631 </td>
   <td style="text-align:right;"> 7.016 </td>
   <td style="text-align:right;"> 97.5 </td>
   <td style="text-align:right;"> 1.40 </td>
   <td style="text-align:right;"> 0.92 </td>
   <td style="text-align:right;"> 1.20 </td>
   <td style="text-align:right;"> 1.28 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 2.96 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 10.100 </td>
   <td style="text-align:right;"> 117.3 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 46 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0592 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50.0 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.631 </td>
   <td style="text-align:right;"> 6.216 </td>
   <td style="text-align:right;"> 100.0 </td>
   <td style="text-align:right;"> 1.38 </td>
   <td style="text-align:right;"> 0.95 </td>
   <td style="text-align:right;"> 1.36 </td>
   <td style="text-align:right;"> 0.99 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 9.53 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 9.800 </td>
   <td style="text-align:right;"> 117.3 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 25 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0609 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 50.0 </td>
   <td style="text-align:right;"> 48.10 </td>
   <td style="text-align:right;"> 0.668 </td>
   <td style="text-align:right;"> 5.875 </td>
   <td style="text-align:right;"> 89.6 </td>
   <td style="text-align:right;"> 1.13 </td>
   <td style="text-align:right;"> 1.01 </td>
   <td style="text-align:right;"> 1.25 </td>
   <td style="text-align:right;"> 1.12 </td>
   <td style="text-align:right;"> 19.8 </td>
   <td style="text-align:right;"> 8.88 </td>
   <td style="text-align:left;"> NO </td>
   <td style="text-align:right;"> 10.800 </td>
   <td style="text-align:right;"> 117.3 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 57 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0645 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 23.7 </td>
   <td style="text-align:right;"> 40.59 </td>
   <td style="text-align:right;"> 0.489 </td>
   <td style="text-align:right;"> 5.412 </td>
   <td style="text-align:right;"> 9.8 </td>
   <td style="text-align:right;"> 3.68 </td>
   <td style="text-align:right;"> 3.48 </td>
   <td style="text-align:right;"> 3.67 </td>
   <td style="text-align:right;"> 3.53 </td>
   <td style="text-align:right;"> 21.4 </td>
   <td style="text-align:right;"> 29.55 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 5.674 </td>
   <td style="text-align:right;"> 111.9 </td>
   <td style="text-align:left;"> None </td>
   <td style="text-align:right;"> 21 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0561 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 48.8 </td>
   <td style="text-align:right;"> 33.97 </td>
   <td style="text-align:right;"> 0.647 </td>
   <td style="text-align:right;"> 6.208 </td>
   <td style="text-align:right;"> 91.5 </td>
   <td style="text-align:right;"> 2.55 </td>
   <td style="text-align:right;"> 2.04 </td>
   <td style="text-align:right;"> 2.39 </td>
   <td style="text-align:right;"> 2.17 </td>
   <td style="text-align:right;"> 27.0 </td>
   <td style="text-align:right;"> 5.91 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 10.076 </td>
   <td style="text-align:right;"> 153.9 </td>
   <td style="text-align:left;"> River </td>
   <td style="text-align:right;"> 24 </td>
   <td style="text-align:left;"> YES </td>
   <td style="text-align:right;"> 0.0551 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
</tbody>
</table></div>

We see that practically all cases with `price = 50` appear as atypical cases. Other outliers come out among the variables where we couldn't establish a clear boundary for atypical cases, e.g., for `teachers > 27 and poor_prop > 29` (as we saw [here](missing-values)).

It will be interesting to see what happens if we adjust the model again, leaving out the outliers detected by analyzing residuals. How do you think that this will affect the model's performance?


```r
lm(
  price ~ age + teachers + poor_prop + room_num + n_hos_beds + n_hot_rooms,
  data = house_prepared %>% filter(!row_number() %in% residuals_outliers)
) %>%
  summary() %>% .$r.squared
```

```
## [1] 0.7768
```

The $R^2$ of the model improves by almost 10 points regarding the model that includes the outliers!

This phenomenon is known as garbage-in garbage-out in the Machine Learning field. If you feed your model with poor quality data, you'll get a low-quality model. The best thing you can do if you want to use your model for prediction is retraining it without these noisy observations.

# Takeaways

This is what we have learned in this blog post:

+ Linear Regression is relatively simple, and so, it is more explainable than other models.
+ Be aware of the outliers. Give the data processing step the care that it deserves. Recheck them when you go through the validation of the model's assumptions.
+ Check the model assumptions (independent variable or residuals normally distributed and homoscedasticity) to ensure that what you are doing is right.
+ Test different variable combinations to see how you can improve your model's performance.

That's it for this post. I hope you have enjoyed it. If you have any comments or suggestions, do not hesitate to contact me!
