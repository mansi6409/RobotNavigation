from stable_baselines3 import PPO
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo

import wandb
from wandb.integration.sb3 import WandbCallback

gym.register_envs(highway_env)

import google.generativeai as genai

genai.configure(api_key="AIzaSyDiQsXdUrKrqOguAE_dTBMqzBjvv4ip0kc")
gemini_model = genai.GenerativeModel("gemini-1.5-flash-002")
LLM_PROMPT = """ You are to act as a reinforcement learning correction agent in a Highway RL environment to to facilitate the self-driving capablities of a vehicle.

To help you understand better:
The OBSERVATION is an array that describes a list of 
 nearby vehicles by a set of features of size, listed in the "features" configuration field.

For instance:
Vehicle x   y   vx  vy
ego-vehicle 0.05    0.04    0.75    0
vehicle 1   -0.15   0   -0.15   0
vehicle 2   0.08    0.04    -0.075  0
...
vehicle V   0.172   0.065   0.15    0.025

- This is configured to normalize=True (default), the observation is normalized within a fixed range, which gives for the range [100, 100, 20, 20]:

- This is also configured with absolute=False, the coordinates are relative to the ego-vehicle, except for the ego-vehicle which stays absolute.

Features-
x: World offset of ego vehicle or offset to ego vehicle on the x axis.
y: World offset of ego vehicle or offset to ego vehicle on the y axis.
vx: Velocity on the x axis of vehicle.
vy: Velocity on the y axis of vehicle.

Also, the actions [ACTION SPACE] you can take in this environment are as follows:
0: 'LANE_LEFT',
1: 'IDLE',
2: 'LANE_RIGHT',
3: 'FASTER',
4: 'SLOWER'
    
Thus the Action space is {action_space}.

Given the following current OBSERVATION:
{observation}

This was the action recommended by the RL agent we're training: {agent_action}

If the recommended action is correct, RETURN THE ACTION AS IS.
Else, reason it out and then return the most suitable action to take to avoid a collision of any sort.

Return ONLY THE RECOMMENDED ACTION FROM THE YOU PICK, NOTHING ELSE.
"""


class LLMActionBiasWrapper(gym.Wrapper):
    def __init__(self, env, model, llm_bias_strength=0.7):
        super().__init__(env)
        self.model = model
        self.llm_bias_strength = llm_bias_strength
        self.current_observation = None

    def reset(self, **kwargs):
        # Capture the initial observation when the environment is reset
        self.current_observation = self.env.reset(**kwargs)
        return self.current_observation

    def step(self, action):
        # Get LLM's action suggestion based on the current observation
        llm_suggested_corrected_action = self.get_llm_suggested_action(self.current_observation, action)

        action = llm_suggested_corrected_action

        self.current_observation, reward, done, truncated, info = self.env.step(action)

        return self.current_observation, reward, done, truncated, info

    def get_llm_suggested_action(self, observation, action):
        # Prompt Gemini model for action recommendation based on observation
        prompt = LLM_PROMPT.format(action_space=self.action_space, observation=observation, agent_action=action)
        llm_response = gemini_model.generate_content(prompt).text
        # Parse llm_response to extract the suggested action
        # print("LLM response:", llm_response)
        try:
          suggested_action = int(llm_response)
          # Check if the suggested action is within the valid action space bounds
          if suggested_action in range(self.action_space.n):
              return suggested_action
          else:
              print(f"Warning: Suggested action {suggested_action} is out of bounds for the action space.")
              return self.env.action_space.sample()  # Fallback to a random action if invalid
        except ValueError:
            print(f"Warning: Unable to parse LLM response '{llm_response}' as an integer action.")
            return self.env.action_space.sample()  # Fallback to a random action if parsing fails

    def modify_action(self, action, llm_suggested_action):
        # Bias action selection towards the LLM suggestion
        if action != llm_suggested_action:
            action = int((1 - self.llm_bias_strength) * action + self.llm_bias_strength * llm_suggested_action)
        return action

# Initialize wrapped environment
env = LLMActionBiasWrapper(gym.make('intersection-v0',render_mode='rgb_array', max_episode_steps=200, # 'highway-fast-v0'
                                    config={
                                              "observation": {
                                                  "type": "OccupancyGrid",
                                                  "vehicles_count": 15,
                                                  "features": ["x", "y", "vx", "vy"],
                                                  "features_range": {
                                                      "x": [-100, 100],
                                                      "y": [-100, 100],
                                                      "vx": [-20, 20],
                                                      "vy": [-20, 20]
                                                  },
                                                  "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                                                  "grid_step": [5, 5],
                                                  "absolute": False
                                              }
                                          }
                                      ), model="Gemini-API-Model")


# Train policy using PPO with action-biasing
# env = RecordVideo(env, video_folder='./videos/merge', episode_trigger=lambda e: e % 10 == 0)
# env.metadata["render_fps"] = 30
from stable_baselines3.common.logger import configure

# Set up SB3 logger to use Tensorboard
log_dir = "./logs"
new_logger = configure(log_dir, ["stdout", "tensorboard"])

model = PPO("MlpPolicy", env, n_steps=200, batch_size=200,  verbose=1)
model.set_logger(new_logger)

# wandb.tensorboard.patch(root_logdir=r'C:\Users\anupa\Documents\CSCI-513 Project\logs')

run = wandb.init(
    project="highway-rl-LLM-Bias",
    config=model.get_parameters(),
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

model.learn(total_timesteps=2000, callback=WandbCallback(verbose=2,))

run.finish()