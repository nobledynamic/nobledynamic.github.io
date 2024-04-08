---
title: "Fabric Madness: Feature Engineering with pyspark"
summary: "In this series of posts titled Fabric Madness, we're going to be diving deep into some of the most interesting features of Microsoft Fabric, for an end-to-end demonstration of how to train and use a machine learning model."
date: 2024-04-08T09:37:43Z
draft: true
showAuthor: true
authors:
  - "martimchaves"
tags:
  - "Microsoft Fabric"
  - "Spark"
series: ["Fabric Madness"]
series_order: 2
---

## Introduction

Feature engineering is a crucial part of the development lifecycle for any Machine Learning (ML) systems. It is a step in the development cycle where raw data is processed to better represent its underlying structure and provide additional information that enhance our ML models. Feature engineering is both an art and a science. Even though there are specific steps that we can take to create good features, sometimes, it is only through experimentation that good results are achieved. Good features are crucial in guaranteeing a good system performance.

As datasets grow exponentially, traditional feature engineering may struggle with the size of very large datasets. This is where PySpark can help - as it is a scalable and efficient processing platform for massive datasets. A great thing about Fabric is that it makes using PySpark easy!

In this post, we'll be going over:
- How does PySpark Work?
- Basics of PySpark
- Feature Engineering in Action

By the end of this post, hopefully you'll feel comfortable carrying out feature engineering with PySpark in Fabric. Let's get started!

## How does PySpark Work?

Spark is a distributed computing system that allows for the processing of large datasets with speed and efficiency across a cluster of machines. It is built around the concept of a Resilient Distributed Dataset (RDD), which is a fault-tolerant collection of elements that can be operated on in parallel. RDDs are the fundamental data structure of Spark, and they allow for the distribution of data across a cluster of machines.

PySpark is the Python API for Spark. It allows for the creation of Spark DataFrames, which are similar to Pandas DataFrames, but with the added benefit of being distributed across a cluster of machines. PySpark DataFrames are the core data structure in PySpark, and they allow for the manipulation of large datasets in a distributed manner.

At the core of PySpark is the `SparkSession` object, which is what fundamentally interacts with Spark. This `SparkSession` is what allows for the creation of DataFrames, and other functionalities. Note that, when running a Notebook in Fabric, a `SparkSession` is automatically created for you, so you don't have to worry about that.

Having a rough idea of how PySpark works, let's get to the basics.

## Basics of PySpark

Although Spark DataFrames may remind us of Pandas DataFrames due to their similarities, the syntax when using PySpark can be a bit different. In this section, we'll go over some of the basics of PySpark, such as reading data, combining DataFrames, selecting columns, grouping data, joining DataFrames, and using functions.

### Reading data

As mentioned in the previous post of this series, the first step is usually to create a Lakehouse and upload some data. Then, when creating a Notebook, we can attach it to the created Lakehouse, and we'll have access to the data stored there.

PySpark Dataframes can read various data formats, such as CSV, JSON, Parquet, and others. Our data is stored in CSV format, so we'll be using that, like in the following code snippet:

```python
# Read women's data
w_data = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(f"Files/WNCAATourneyDetailedResults.csv")
    .cache()
)
```

In this code snippet, we're reading the detailed results data set of the final women's basketball college tournament matches. Note that the `"header"` option being true means that the names of the columns will be derived from the first row of the CSV file. The `inferSchema` option tells Spark to guess the data types of the columns - otherwise they would all be read as strings. `.cache()` is used to keep the DataFrame in memory.

If you're coming from Pandas, you may be wondering what the equivalent of `df.head()` is for PySpark - it's `df.show(5)`. The default for `.show()` is the top 20 rows, hence the need to specifically select 5.

### Combining DataFrames

Combining DataFrames can be done in multiple ways. The first we will look at is a union, where the columns are the same for both DataFrames:

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

Here, `unionByName` joins the two DataFrames by matching the names of the columns. Since both the women's and the men's *detailed match results* have the same columns, this is a good approach. Alternatively, there's also `union`, which combines two Dataframes, matching column positions.

### Selecting Columns

Selecting columns from a DataFrame in PySpark can be done using the `.select()` method. We just have to indicate the name or names of the columns that are relevant as a parameter.

```python
# Selecting a single column
w_scores = w_data.select("WScore")

# Selecting multiple columns
teamid_w_scores = w_data.select("WTeamID", "WScore")
```

Here's the output for `w_scores.show(5)`:
```
+------+
|Season|
+------+
|  2010|
|  2010|
|  2010|
|  2010|
|  2010|
+------+
only showing top 5 rows
```

The columns can also be renamed when being selected using the `.alias()` method:

```python
winners = w_data.select(
    w_data.WTeamID.alias("TeamID"),
    w_data.WScore.alias("Score"))
```

### Grouping Data

Grouping allows us to carry out certain operations for the groups that exist within the data and is usually combined with a aggregation functions. We can use `.groupBy()` for this:

```python
# Grouping and aggregating
winners_average_scores = winners.groupBy("TeamID").avg("Score")
```

In this example, we are grouping by `"TeamID"`, meaning we're considering the groups of rows that have a distinct value for `"TeamID"`. For each of those groups, we're calculating the average of the `"Score"`. This way, we get the average score for each team.

Here's the output of `winners_average_scores.show(5)`, showing the average score of each team:

```
+------+-----------------+
|TeamID|       avg(Score)|
+------+-----------------+
|  3125|             68.5|
|  3345|             74.2|
|  3346|79.66666666666667|
|  3376|73.58333333333333|
|  3107|             61.0|
+------+-----------------+
```

### Joining Data

Joining two DataFrames can be done using the `.join()` method. Joining is essentially extending the DataFrame by adding the columns of one DataFrame to another.

```python
# Joining on Season and TeamID
final_df = matches_df.join(stats_df, on=['Season', 'TeamID'], how='left')
```

In this example, both `stats_df` and `matches_df` were using `Season` and `TeamID` as unique identifiers for each row. Besides `Season` and `TeamID`, `stats_df` has other columns, such as statistics for each team during each season, whereas `matches_df` has information about the matches, such as date and location. This operation allows us to add those interesting statistics to the matches information!

### Functions

There are several functions that PySpark provides that help us transform DataFrames. You can find the full list [here](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html).

Here's an example of a simple function:
```python
from pyspark.sql import functions as F

w_data = w_data.withColumn("HighScore", F.when(F.col("Score") > 80, "Yes").otherwise("No"))
```

In the code snippet above, a `"HighScore"` column is created when the score is higher than 80. For each row in the `"Score"` column (indicated by the `.col()` function), the value `"Yes"` is chosen for the `"HighScore"` column if the `"Score"` value is larger than 80, determined by the `.when()` function. `.otherwise()`, the value chosen is `"No"`.

## Feature Engineering in Action

### Regular Season Statistics

Now that we have a basic understanding of PySpark and how it can be used, let's go over how the regular season statistics features were created. These features were then used as inputs into our machine learning model to try to predict the outcome of the final tournament games.

The starting point was a DataFrame, `regular_data`, that contained match by match statistics for the *regular seasons*, which is the United States College Basketball Season that happens from November to March each year.

Each row in this DataFrame contained the season, the day the match was held, the ID of team 1, the ID of team 2, and other information such as the location of the match. Importantly, it also contained statistics for *each team* for that *specific match*, such as `"T1_FGM"`, meaning the Field Goals Made (FGM) for team 1, or `"T2_OR"`, meaning the Offensive Rebounds (OR) of team 2. 

The first step was selecting which columns would be used. These were columns that strictly contained in-game statistics.

```python
# Columns that we'll want to get statistics from
boxscore_cols = [
    'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_DR', 'T1_Ast', 'T1_Stl', 'T1_PF', 
    'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_DR', 'T2_Ast', 'T2_Stl', 'T2_PF'
]
```

If you're interested, here's what each statistic's code means:
- FGM: Field Goals Made
- FGA: Field Goals Attempted
- FGM3: Field Goals Made from the 3-point-line
- FGA3: Field Goals Attempted for 3-point-line goals
- OR: Offensive Rebounds. A rebounds is when the ball rebounds from the board when a goal is attempted, not getting in the net. If the team that *attempted* the goal gets possession of the ball, it's called an "Offensive" rebound. Otherwise, it's called a "Defensive" Rebound.
- DR: Defensive Rebounds
- Ast: Assist, a pass that led directly to a goal
- Stl: Steal, when the possession of the ball is stolen
- PF: Personal Foul, when a player makes a foul


From there, a dictionary of *aggregation expressions* was created. Basically, for each column name in the previous list of columns, a function was stored that would calculate the mean of the column, and rename it, by adding a suffix, `"mean"`.


```python
from pyspark.sql import functions as F
from pyspark.sql.functions import col  # select a column

agg_exprs = {col: F.mean(col).alias(col + 'mean') for col in boxscore_cols}
```

Then, the data was grouped by `"Season"` and `"T1_TeamID"`, and the aggregation functions of the previously created dictionary were used as the argument for `.agg()`. Note that `(*agg_exprs.values())` is the same as `(F.mean('T1_FGM').alias('T1_FGMmean'), F.mean('T1_FGA').alias('T1_FGAmean'), ..., F.mean('T2_PF').alias('T2_PFmean'))`.

```python
season_statistics = regular_data.groupBy(["Season", "T1_TeamID"]).agg(*agg_exprs.values())
```

Note that the grouping was done by season and the **ID of team 1** - this means that `"T2_FGAmean"`, for example, will actually be the mean of the Field Goals Attempted made by the **opponents** of T1, not necessarily of a specific team. So, we actually need to rename the columns that are something like `"T2_FGAmean"` to something like `"T1_opponent_FGAmean"`.

```python
# Rename columns for T1
for col in boxscore_cols:
    season_statistics = season_statistics.withColumnRenamed(col + 'mean', 'T1_' + col[3:] + 'mean') if 'T1_' in col \
        else season_statistics.withColumnRenamed(col + 'mean', 'T1_opponent_' + col[3:] + 'mean')
```

At this point, it's important to mention that the `regular_data` DataFrame actually has **two** rows per each match that occurred. This is so that both teams can be "T1" and "T2", for each match. This little "trick" is what makes these statistics useful.

Note that we "only" have the statistics for "T1". We "need" the statistics for "T2" as well - "need" in quotations because there are no new statistics being calculated. We just need the same data, but with the columns having different names, so that for a match with "T1" and "T2", we have statistics for both T1 and T2. So, we created a mirror DataFrame, where, instead of "T1...mean" and "T1_opponent_...mean", we have "T2...mean" and "T2_opponent_...mean". This is important because, later on, when we're joining these regular season statistics to tournament matches, we'll be able to have statistics for both team 1 **and** team 2.

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

### Elo Ratings

First created by [Arpad Elo](https://en.wikipedia.org/wiki/Elo_rating_system), Elo is a rating system for zero-sum games (games where one player wins and the other loses), like basketball. With the Elo rating system, each team has an Elo rating, a value that generally conveys the team's quality. At first, every team has the same Elo, and whenever they win, their Elo increases, and when they lose, their Elo decreases. A key characteristic of this system is that this value increases more with a win against a strong opponent than with a win against a weak opponent. Thus, it can be a very useful feature to have!

We wanted to capture the Elo rating of a team at the end of the regular season, and use that as feature for the tournament. To do this, we calculated the Elo for each team on a per match basis. To calculate Elo for this feature, we found it more straightforward to use Pandas.

Central to Elo is calculating the expected score for each team. It can be described in code like so:

```python
# Function to calculate expected score
def expected_score(ra, rb):
    # ra = rating (Elo) team A
    # rb = rating (Elo) team B
    # Elo function
    return 1 / (1 + 10 ** ((rb - ra) / 400))
```

Considering a team A and a team B, this function computes the expected score of team A against team B.

For each match, we would update the teams' Elos. Note that the location of the match also played a part - winning at home was considered less impressive than winning away.

```python
# Function to update Elo ratings, keeping T1 and T2 terminology
def update_elo(t1_elo, t2_elo, location, T1_Score, T2_Score):
    expected_t1 = expected_score(t1_elo, t2_elo)
    expected_t2 = expected_score(t2_elo, t1_elo)
    
    actual_t1 = 1 if T1_Score > T2_Score else 0
    actual_t2 = 1 - actual_t1

    # Determine K based on game location
    # The larger the K, the bigger the impact
    # team1 winning at home (location=1) less impressive than winning away (location = -1)
    if actual_t1 == 1:  # team1 won
        if location == 1:
            k = 20
        elif location == 0:
            k = 30
        else:  # location = -1
            k = 40
    else:  # team2 won
        if location == 1:
            k = 40
        elif location == 0:
            k = 30
        else:  # location = -1
            k = 20
    
    new_t1_elo = t1_elo + k * (actual_t1 - expected_t1)
    new_t2_elo = t2_elo + k * (actual_t2 - expected_t2)
    
    return new_t1_elo, new_t2_elo
```

To apply the Elo rating system, we iterated through each season's matches, initializing teams with a base rating and updating their ratings match by match. The final Elo available for each team in each season will, hopefully, be a good descriptor of the team's quality.

```python

def calculate_elo_through_seasons(regular_data):

    # For this feature, using Pandas
    regular_data = regular_data.toPandas()
    
    # Set value of initial elo
    initial_elo = 1500

    # DataFrame to collect final Elo ratings
    final_elo_list = []

    for season in sorted(regular_data['Season'].unique()):
        print(f"Season: {season}")
        # Initialize elo ratings dictionary
        elo_ratings = {}

        print(f"Processing Season: {season}")
        # Get the teams that played in the season
        season_teams = set(regular_data[regular_data['Season'] == season]['T1_TeamID']).union(set(regular_data[regular_data['Season'] == season]['T2_TeamID']))
        
        # Initialize season teams' Elo ratings
        for team in season_teams:
            if (season, team) not in elo_ratings:
                elo_ratings[(season, team)] = initial_elo

        # Update Elo ratings per game
        season_games = regular_data[regular_data['Season'] == season]
        for _, row in season_games.iterrows():
            t1_elo = elo_ratings[(season, row['T1_TeamID'])]
            t2_elo = elo_ratings[(season, row['T2_TeamID'])]

            new_t1_elo, new_t2_elo = update_elo(t1_elo, t2_elo, row['location'], row['T1_Score'], row['T2_Score'])
            
            # Only keep the last season rating
            elo_ratings[(season, row['T1_TeamID'])] = new_t1_elo
            elo_ratings[(season, row['T2_TeamID'])] = new_t2_elo

        # Collect final Elo ratings for the season
        for team in season_teams:
            final_elo_list.append({'Season': season, 'TeamID': team, 'Elo': elo_ratings[(season, team)]})

    # Convert list to DataFrame
    final_elo_df = pd.DataFrame(final_elo_list)

    # Separate DataFrames for T1 and T2
    final_elo_t1_df = final_elo_df.copy().rename(columns={'TeamID': 'T1_TeamID', 'Elo': 'T1_Elo'})
    final_elo_t2_df = final_elo_df.copy().rename(columns={'TeamID': 'T2_TeamID', 'Elo': 'T2_Elo'})

    # Convert the pandas DataFrames back to Spark DataFrames
    final_elo_t1_df = spark.createDataFrame(final_elo_t1_df)
    final_elo_t2_df = spark.createDataFrame(final_elo_t2_df)

    return final_elo_t1_df, final_elo_t2_df
```

Ideally, we wouldn't calculate Elo changes on a match-by-match basis to determine each team's final Elo for the season. However, we couldn't come up with a better approach. Do you have any ideas? If so, let us know!

### Value Added

The feature engineering steps demonstrated show how we can transform raw data - regular season statistics - into valuable information with predictive power. It is reasonable to assume that a team's performance during the regular season is indicative of its potential performance in the final tournaments. By calculating the mean of observed match-by-match statistics for both the teams and their opponents, along with each team's Elo rating in their final match, we were able to create a dataset suitable for modelling. Then, models were trained to predict the outcome of tournament matches using these features, among others developed in a similar way. With these models, we only need the two team IDs to look up the mean of their regular season statistics and their Elos to feed into the model and predict a score!

## Conclusion

In this post, we looked at some of the theory behind Spark and PySpark, how that can be applied, and a concrete practical example. We explored how feature engineering can be done in the case of sports data, creating regular season statistics to use as features for final tournament games. Hopefully you've found this interesting and helpful - happy feature engineering!
