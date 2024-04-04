---
title: "Fabric Madness: Feature Engineering with pyspark"
summary: "In this series of posts titled Fabric Madness, we're going to be diving deep into some of the most interesting features of Microsoft Fabric, for an end-to-end demonstration of how to train and use a machine learning model."
date: 2024-03-27T11:37:43Z
draft: true
showAuthor: true
authors:
  - "martimchaves"
  - "rogernoble"
tags:
  - "Microsoft Fabric"
  - "Spark"
series: ["Fabric Madness"]
series_order: 2
---

## Introduction

Feature Engineering is crucial in the development of Machine Learning (ML) systems. It is a step in the development cycle where raw data is processed to better represent the underlying structure of the problem at hand. As the name dictates, the aim is to create, engineer, something that represents the data - features! It is both an art and a science. Even though there are specific steps that we can take to create good features, sometimes, it is only through experimentation that good results are achieved. Good features are crucial in guaranteeing a good system performance.

As datasets grow exponentially, traditional feature engineering may struggle with the size of very large datasets. This is where PySpark can help - it can allow for a scalable and efficient processing of massive datasets. Another great thing about Fabric is that it makes using PySpark is very straighforward.

In this post, we'll be going over:
- How does PySpark Work?
- Basics of PySpark
- Feature Engineering in Action

By the end of this post, hopefully you'll feel confortable carrying out feature engineering with PySpark in Fabric!

## How does PySpark Work?

PySpark is the Python API for Spark. So, it makes sense to first go into what Spark is. It is a tool to process and analyse large amounts of data quickly. Put simply, it distributes data across many machines, allowing us to process data in parallel, instead of running processing steps in a single machine. These machines, that process the data, form what is called a "Spark Cluster". PySpark acts as an interface for what Spark provides, such as Spark DataFrames.

Spark DataFrames are similar to Pandas DataFrames, as they're both Tables. But, there's a big difference - under the hood, Spark takes care of distributing the Spark DataFrame's rows across the available machines in a Spark Cluster.

At the core of PySpark is the `SparkSession` object, which is what fundamentally interacts with Spark. This `SparkSession` is what allows for the creation of DataFrames, and other functionalities. Note that, when running a Notebook in Fabric, a SparkSession is automatically created!

Having a rough idea of how PySpark works, let's get to the basics.

## Basics of PySpark

Although Spark DataFrames may remind us of Pandas DataFrames due to their similarities, the syntax when using Spark can be a bit different. Sometimes, it sort of resembles SQL code!

### Reading data

As mentioned in the previous post of this series, the first step when using any tool in Fabric is to create a Lakehouse and upload some data. Then, when creating a Notebook, we can attach it to the created Lakehouse, and we'll have access to the data stored there.

PySpark can read various data formats, such as CSV, JSON, Parquet, and other. Reading data into a DataFrame can be done like so:

```python
# Read women's data
w_data = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(f"Files/WNCAATourneyDetailedResults.csv")
    .cache()
)
```

In this code snippet, we're reading the Detailed Results of the final women's basketball college tournament matches. It's not needed to either import or start the Spark session, that's done automatically. Note that the "header" option being true means that the names of the columns will be derived from the first row of the CSV file. The "inferSchema" option tells Spark to guess the data types of the columns - otherwise they would all be read as strings. `.cache()` is used to keep the DataFrame in memory.

### Combining DataFrames

Combining DataFrames can be done in multiple ways. Let's look at one where the columns are the same for both DataFrames:

```python
# Read women's data
...

# Read men's data
m_data = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(f"Files/MNCAATourneyDetailedResults.csv")
    .cache()
)

# Combine (union) the DataFrames
combined_results = m_data.unionByName(w_data)
```

Here, `unionByName` joins the two DataFrames by matching the names of the column. There's also `union`, which combines two Dataframes, matching column positions.

### Selecting Columns

Selecting columns from a DataFrame in PySpark can be done using the `.select()` method. We just have to indicate the name or names of the columns that are relevant as a parameter.

```python
# Selecting a single column
team_names = w_data.select("WScore")

# Selecting multiple columns
team_info = w_data.select("WTeamID", "WScore")
```

The columns can also be renamed when being selected using the `.alias()` method:

```python
winners = w_data.select(
    w_data.WTeamID.alias("TeamID"),
    w_data.WScore.alias("Score"))
```

### Grouping Data

Grouping allows us to carry out certain operations for the groups that exist within the data. We can use `.groupBy()` for this:

```python
# Grouping and aggregating
winners_average_scores = winners.groupBy("TeamID").avg("Score")
```

In this example, we got the average score for each team!

### Joining Data

Joining two DataFrames can be done using the `.join()` method.

```python
# Joining on Season and TeamID
final_df = matches_df.join(stats_df, on=['Season', 'TeamID'], how='left')
```

In this example, both `stats_df` and `matches_df` were using `Season` and `TeamID` as unique identifiers. Besides `Season` and `TeamID`, `stats_df` has other columns, such as statistics for each team during each season, whereas `matches_df` has information about the matches, such as date and location. This operation allows us to add those interesting statistics to the matches information!

### Functions

There are several functions that PySpark provides that help us transform DataFrames. You can find the full list [here](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html).

Here's an example of a simple function:
```python
from pyspark.sql import functions as F

w_data = w_data.withColumn("HighScore", F.when(F.col("Score") > 80, "Yes").otherwise("No"))
```

In the code snippet above, a "HighScore" column is created when the score is higher than 80. For each row in the "Score" column (indicated by the `.col()` function), the value "Yes" is chosen for the "HighScore" column if the "Score" value is larger than 80, determined by the `.when()` function. `.otherwise()`, the value chosen is "No".

## Feature Engineering in Action

Now that we have a basic understanding of PySpark and how it can be used, let's go over how the Regular Season Statistics features were created! These features were then used to try to predict the outcome of the Final Tournament games.

The first step was selecting which columns would be used. These were columns that contained in-game statistics such as Field Goals Made (FGM) and Offensive Rebounds (OR).

```python
# Columns that we'll want to get statistics from
boxscore_cols = [
    'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF', 
    'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',  
    'PointDiff'
]
```
Note that in the detailed results, for each match, there were statistics for Team 1 (T1) and Team 2 (T2). For each game there are two rows, so that both teams, for each game, can be T1 and T2. This will be important later on.

From there, a dictionary of *aggregation expressions* was created. Basically, for each column name in the previous list of columns, a function was stored that would calculate the mean of the column, and add rename it, by adding a suffix, "mean".


```python
from pyspark.sql import functions as F
from pyspark.sql.functions import col  # select a column

agg_exprs = {col: F.mean(col).alias(col + 'mean') for col in boxscore_cols}
```

Then, the data was grouped by "Season" and "TeamID", and the aggregation functions of the previously created dictionary were used as the argument for the `.agg()`. Note that `(*agg_exprs.values())` is the same as `(F.mean('T1_FGM').alis('T1_FGMmean'), F.mean('T1_FGA').alis('T1_FGAmean'), ...)`.

```python
season_statistics = regular_data.groupBy(["Season", "T1_TeamID"]).agg(*agg_exprs.values())
```

Note that the grouping was done by "Season" and "**T1**_TeamID" - this means that "T2_FGAmean", for example, will actually be the mean of the Field Goals Assists made by the opponents of the T1, not necessarily of a specific team. So, we actually need to rename the columns that have "T2_FGAmean" to something like "T1_opponent_FGAmean".

```python
# Rename columns for T1
for col in boxscore_cols:
    season_statistics = season_statistics.withColumnRenamed(col + 'mean', 'T1_' + col[3:] + 'mean') if 'T1_' in col \
        else season_statistics.withColumnRenamed(col + 'mean', 'T1_opponent_' + col[3:] + 'mean')
```

Finally, at this point we "only" have the statistics for "T1". We "need" the statistics for "T2" - "need" in quotations because there are no new statistics being calculated. We just need the same data, but with the columns having different names, so that for a match with "T1" and "T2", we have statistics for both T1 and T2. So we create a mirror DataFrame, where instead of "T1...mean" and "T1_opponent_...mean", we have "T2...mean" and "T2_opponent_...mean":

```python
season_statistics_T2 = season_statistics.select(
    *[F.col(col).alias(col.replace('T1_opponent_', 'T2_opponent_').replace('T1_', 'T2_')) if col not in ['Season'] else F.col(col) for col in season_statistics.columns]
)
```

Now, there are two DataFrames, with season statistics for "both" T1 and T2. Since the final dataframe will contain the "Season", the "T1TeamID" and the "T2TeamID", we can join these newly created features with a join!

```python
tourney_df = tourney_df.join(season_statistics, on=['Season', 'T1_TeamID'], how='left')
tourney_df = tourney_df.join(season_statistics_T2, on=['Season', 'T2_TeamID'], how='left')
```

## Conclusion

In this post, we've looked at the some of the theory behind Spark and PySpark, how that can be applied, and a concrete practical example. We've explored how feature engineering can be done in the case of sports data, creating regular season statistics to use as features for the final tournament games. Hopefully you've found this interesting and helpful - happy feature engineering!