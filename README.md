# Briscola-py

RL for briscola, using PPO

Play against the 4 player model on https://briscola.io !

Export model for other platforms: https://stable-baselines3.readthedocs.io/en/master/guide/export.html#export-to-onnx

# Achievements

## Best model

The best model trained here was in the 1v1 case. When briscola is played 1v1, the game becomes zero sum, making it better suited for PPO. The model achieved a clear advantage over random play.

<img width="474" height="744" alt="image" src="https://github.com/user-attachments/assets/014f90ad-e14f-4e14-ba7b-5e3b51100725" />

The 4-player model slightly outperforms random play from each starting position, but not by a significant margin. I have observed it sort of throwing away points in early stages of the game, and consistently holding batons until the end of the game regardless of what the briscola suit is. 

## Takeaways

Basing the reward function on the number of points gained per tern was much more effective than a simple 1st/2nd/3rd place reward. 
Providing a negative reward when giving up points unnecessarily also made the model much more successful.

CFM may perform much better for the 4-player case given the similarity to poker.
