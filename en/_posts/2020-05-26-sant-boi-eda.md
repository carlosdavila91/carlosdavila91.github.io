---
title: Exploratory Data Analysis
author: Carlos Dávila Luelmo
date: 2020-05-26
lang: en
lang-ref: sant-boi-eda
tags:
  - sustainability
  - construction
  - energy
  - Spain
  - Catalonia
  - Sant Boi de Llobregat
---

Univariate Exploratory Data Analysis to understand the attributes gathered for the project.

## Exploratory Data Analysis (EDA) - Discrete Variables

### District

The study is oriented to the districts of [Marianao](https://www.google.es/maps/place/Marianao/@41.3479877,2.0263476,1045m/data=!3m1!1e3!4m5!3m4!1s0x12a49b822306e79d:0x3297c05bdd8c8ad1!8m2!3d41.3487888!4d2.0287207) and [Vinyets](https://www.google.es/maps/@41.3374306,2.0438772,919m/data=!3m1!1e3), according to the conditions established by the municipality.

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/barrio.png">
</p>

MARIANAO| VINYETS
---- | ----
499 | 315

### Decade

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/decada.png">
</p>

Before 1899| From 1900 to 1940| From 1941 to 1960| From 1961 to 1970| From 1971 to 1980
---|---|---|---|---
19| 132| 90| 312 | 261

This city suffered a vast growth during periods of strong internal migration in the interior of the country. Notice that in the second bar in the chart there are four decades aggregated.

### Orientation of Buildings

Orientation is one of the most important factors when it comes to energy efficiency of households.

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/orientacion.png">
</p>

N| S|   E|   W| SE |NE| NW| SW
---|---|---|---|---|---|---|---
134| 131| 109| 106| 98| 86| 79| 71

The most frequent orientations are the four main cardinal points.

Regarding Bioclimatic Architectonic criteria, each orientation has its own pros and cons for this latitude (41º) and the temperate Mediterranean climate:

* SE, S and SW are preferred. Those can help to use less heating in winter by taking the direct sun radiation during practically all day in winter. This radiation can be avoided in summer easily with horizontal sun protections.
* N, NE and NW does not have the advantage of taking radiation in winter, although they give a nice indirect illumination all the year. NE and NW may produce not desirable radiation in summer during sunrise and sunset. The common strategy to solve this are vertical sun protections.
* E and W are the less desirable ones. You may have to expend a lot of air conditioning in summer.

These specificities are very relative in an urban environment with buildings located near each other and other elements as trees, that may cast shadows on buildings.

### Number of dwellings

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/viviendas.png">
</p>

Detached |From 2 to 4 dwellings| From 5 to 9 dwellings| From 10 to 19 dwellings| From 20 to 39 dwellings| More than 40 dwellings
---|---|---|---|---|---
261|182| 111|155|82|22

This variable should be taken into account in terms of how many people is going to be affected by a refurbishment oriented to building energy efficiency.

From another point of view, it would determine the ratio "money invested" over "people affected" of the operation. This may take single family households out of the scope of the strategy.

### Number of floors

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/plantas.png">
</p>

1 floor|   2 floors|   3 floors|  4 floors|   5 floors|  6 floors|  7 floors|  8 floors
---|---|---|---|---|---|---|---
 67| 204| 107| 84| 207| 84| 50| 11

This variable is, again, one of the most important ones regarding to the investment that will be made in buildings energy efficiency.

It will be clear that single-storey buildings are candidates to stay out of the building renovation strategy.

Although it is a numerical variable, it will be considered as a categorical one, due to its ability to distinguish different types of buildings.

### Use of the ground floor

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/pbaja.png">
</p>

 Dwelling| Commercial| Storage| Industrial
 ---|---|---|---
400 | 355| 41| 17

Speaking from an energy point of view, is different to have a home than other types of usages in direct contact to your own household.

For example, in winter, when one dwelling is attached to another there won't be a heat exchange between them, as both will be commonly at the same temperature.

When the adjacent space is intended for a different use than this, it is not guaranteed that it is heated continuously, so temperature exchanges may occur.

### Type of facade

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/fachadas.png">
</p>

F1| F2|  F3
---|---|---
 468| 47| 299

Essentially this nomenclature correspond to specific and commmon construction systems in the context of the study.

* F1: Single sheet of brick, thickness of approximately 30 cm.
* F2: Single sheet of brick, thickness of approximately 15 cm.
* F3: Facade walls with air chamber of 15/10/5 cm

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/Facades.png">
</p>

Each has its associated transmittance value. This physical property affects to the ability of the material system to keep heat or cold inside the home. This is the intuition under the following variables which classify other physical elements of buildings.

### Types of Roof

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/cubiertas.png">
</p>

C1|  C2|  C3|  C4|
---| ---| ---| ---|
161| 427| 141|  85|

Once again, this variable deals with the transmittance in different cases:

* C1: Ventilated flat roof
* C2: Non-ventilated flat roof
* C3: Ventilated sloping roof
* C4: Non-ventilated sloping roof

### Types of facade opening

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/huecos.png">
</p>

H1|  H2|  H3|  H4|  H5|
---| ---| ---| ---| ---|
 26|  28| 471| 284|   5|

In this case the different categories are mainly determined by whether or not they have been renewed and whether they have solar protections. Only windows are considered for the aim of this study.

* H1: Windows with no sign of alteration and without solar protections.
* H2: Renovated windows, without solar protections
* H3: Windows with no sign of alteration, with solar protections
* H4: Renovated windows, with solar protections
* H5: Opaque enclosures: entrance doors, garage doors,...

### Types of party wall

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/medianeras.png">
</p>

0|  M1|  M2|  M3|
---| ---| ---| ---|
308| 255| 243|   8|

Repeatedly here we describe the transmittance of the party wall with several categories. Those which are protected have an air chamber that decreases the amount of energy transferred to the exterior. This occurs when a specific construction system was founded.

* M1: Unprotected party wall
* M2: Protected party wall
* M3: Others

## Exploratory Data Analysis (EDA) - Continuous Variables

The variables represented in the next multivariate plot (credits for [Barret Schloerke](https://github.com/ggobi/ggally/issues/139#issuecomment-176271618) and [user20650](https://stackoverflow.com/a/34517880/11597692)) are the following:

* Roof Surface
* Facade Surface
* Openings Surface
* Surface Touching the Ground

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/continuas.png">
</p>

Strong correlations appear between these physical characteristics of buildings.

When we distinguish buildings by district there are not clear differences, except that buildings in Marianao may be slightly larger.

As we can see, buildings with similar number of heights present similar physical characteristics.

<p align="center">
  <img src="{{site.baseurl}}/images/sant-boi-eda/densidad.png">
</p>

All the information gathered here was taken into account in order to reach the aim of the study developed for the Sant Boi de Llobregat City Council.

[_See the code used for this post here_](https://github.com/carlosdavila91/santboi_eda/tree/master/code)
