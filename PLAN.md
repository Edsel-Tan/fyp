Goal: to be able to predict LoL match outcomes and check if it is profitable

We want to train a model on league matches and check if it is profitable. There are two main sections of this research and further subsections explored

1. Training the model

The model's goal is given a current game state, output a probability of a team to win. There are two stages to this: pre-game and during game.

Pre-game, the model requires
- each player's rating (whether by name or just a raw elo)
- each team's rating (whether by name or just a raw elo)
- recent performance

We need to explore how to use each stats. For example, if we hardcode a player's name e.g. Faker, the model might overfit for worlds games to give an extremely high probability of winning. We also need to figure out how to get each player's stats. 
Liquidpedia is a possible first step. When training the model, ensure no leak forward. 

During game, the model requires
- every pre-game stat
- each player's current kda, cs and gold
- each teams current kda, total gold and map state

Again, we need to explore how to use each stats. We can use image recognition to screenshot livestreams periodically for each stat. For map state, we can use a bit map of the minimap to represent the map state. When training, we need to give a 
realistic latency (i.e. 10:30 game data can only be traded at 10:35 for a 5 second delay) (this might not be entirely correct, we need to test what the actual latency could be).

2. Executing the model

Now with the model, we need to test if it is actually profitable and the trading strategy to employ. Two possible simple strategy

- We buy a fixed amount every X seconds in the direction to shift the market to our predicted probability.
- We predict a second value, our confidence in the probability and only cross the spread if our confidence is deemed to be more profitable than the spread and the deviation from our predicted probability.

You should investigate and execute the research plans as follow:

1. Attempt to pull data for the inputs to the models. Test different constructions of each statistic (how long a player has played etc, what elo measure is used, or do we use win-lose, or head-to-head win-lose)
2. Test different models. One simple model is to ignore everything and just use the gold difference. Train a function that takes in the gold and outputs a value for the probability.
3. Perform significance tests to ensure no overfitting. 
4. Test agaisnt baselines. (Equal weight, follow informed traders, simple model stated above)

In the branch hrt-production, there is instruction on how to connect to RTX 4000 blackwell, if you need to, inform me and I will spin the sandbox up. Report the results in @RESULTS.md
