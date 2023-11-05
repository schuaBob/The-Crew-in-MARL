from crew_gym_env import CustomCrewGymEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import MultiInputActorCriticPolicy as MlpPolicy

# Define a function to create and train the PPO agent
def train_ppo_agent(env_name, total_timesteps):
    # Create the environment
    env = CustomCrewGymEnvironment()

    # Wrap the environment in a vectorized environment (if needed)
    env = DummyVecEnv([lambda: env])

    # Create the PPO agent
    model = PPO(MlpPolicy, env, verbose=1)  # You can choose a different policy and model if needed

    # Train the agent
    model.learn(total_timesteps)

    # Save the trained agent
    model.save("ppo_custom_crew")

if __name__ == "__main__":
    train_ppo_agent("CustomCrewEnvironment", total_timesteps=10000)  # Adjust the total_timesteps as needed
