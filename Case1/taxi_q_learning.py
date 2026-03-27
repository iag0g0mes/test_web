import argparse
import os
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


# ---------------------------
# Viewing with a fixed window
# ---------------------------
class TaxiViewer:
    """
    It maintains a fixed Matplotlib window and updates the image at each step.
    """
    def __init__(self, title="Taxi-v3"):
        plt.ion()  # interactive mode
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        self.im = None
        self.ax.axis("off")
        self.fig.tight_layout()

    def show(self, frame, subtitle=None):
        if subtitle is not None:
            self.ax.set_title(subtitle)

        if self.im is None:
            self.im = self.ax.imshow(frame)
        else:
            self.im.set_data(frame)

        # draw/update without closing
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ---------------------------
# Q-learning training
# ---------------------------
def train_q_learning(
                num_episodes=5000,
                alpha=0.1,
                gamma=0.99,
                epsilon=0.1,
                visualize=False,
                viz_every=200,        # Show every few episodes (if view=True)
                viz_max_steps=200,    # Limit the number of steps for viewing (prevents it from becoming infinite).
                ):

    env = gym.make("Taxi-v3", render_mode="rgb_array")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros([n_states, n_actions])

    cum_rewards = np.zeros([num_episodes])
    total_steps = np.zeros([num_episodes])

    if visualize:
        viewer = TaxiViewer("Taxi - Training")
    else:
        viewer = None


    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        done = False
        cum_reward = 0.0
        steps = 0

        show_this_episode = visualize and (episode % viz_every == 0)

        while not done:
            if random.uniform(0, 1) < epsilon:     # epsilon-greedy with action_mask (when it exists)
                mask = info["action_mask"] 
                if mask is not None:
                    action = env.action_space.sample(mask)  # Sample random action with action mask (exploration)
                else:
                    action = env.action_space.sample()  # Sample random action (exploration)
            else:
                mask = info["action_mask"]
                q_masked = np.where(mask, q_table[state], -np.inf)  # Mask invalid actions with -inf
                action = np.argmax(q_masked) # Select best action from Q-table (exploitation)

            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            old_q_value = q_table[state, action]  # Q value for current state and action    

            if done:             # if episode ended, next state is terminal, so max Q value for next state is 0
                target = reward
            else:  
                next_q_max = np.max(q_table[next_state])  # max Q value for next state
                target = reward + gamma * next_q_max

            new_q_value = (1 - alpha) * old_q_value + alpha * target
            q_table[state, action] = new_q_value  # Update Q-table with new Q value

            cum_reward += reward
            steps += 1

            if show_this_episode and steps <= viz_max_steps:
                frame = env.render()
                viewer.show(frame,
                            subtitle=f"TRAIN ep={episode}/{num_episodes} step={steps} action={action} reward={reward} total={cum_reward:.1f}"
                                )

            state, info = next_state, next_info

        cum_rewards[episode - 1] = cum_reward
        total_steps[episode - 1] = steps

        if episode % 500 == 0:
            last = cum_rewards[max(0, episode-100):episode]
            print(f"Ep {episode:5d} | avg_reward(last100)={float(np.mean(last)):.2f} | steps={steps}")


    env.close()
    if viewer:
        viewer.close()

    # plots
    plt.figure()
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(cum_rewards)
    plt.show()

    plt.figure()
    plt.title("Steps per episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.plot(total_steps)
    plt.show()

    return q_table



# ---------------------------
# Test the policy
# ---------------------------
def test_policy(
            q_table,
            episodes=50,
            visualize=True,
            max_steps=200,
            ):
    
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    # u = env.unwrapped
    if visualize:
        viewer = TaxiViewer("Taxi - Test") 
    else:
        viewer = None

    successes = 0
    truncations = 0
    manual_timeouts = 0
    rewards_all = []
    steps_all = []


    for episode in range(1, episodes + 1):
        state, info = env.reset()
        # print("state:", state, "->", tuple(u.decode(state)))
        done = False
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            mask = info["action_mask"].astype(bool)
            q_masked = np.where(mask, q_table[state], -np.inf)  # Mask invalid actions with -inf
            action = np.argmax(q_masked)  # Select the action with the highest Q value among valid actions
            next_state, reward, terminated, truncated, next_info = env.step(action)   
            # print("next_state:", next_state, "->", tuple(u.decode(next_state))) # (taxi_row, taxi_col, passenger_location, destination)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state
            info = next_info

            if visualize:
                frame = env.render()
                viewer.show(frame,
                        subtitle=f"TEST ep={episode}/{episodes} step={steps} action={action} reward={reward} total={total_reward:.1f}"
                        )
                
        if (not terminated) and (not truncated) and (steps >= max_steps):
            manual_timeouts += 1

        if terminated:
            successes += 1
        elif truncated:
            truncations += 1


        rewards_all.append(total_reward)
        steps_all.append(steps)

        print(
            f"[TEST] ep {episode}: reward={total_reward:.1f} steps={steps} "
            f"terminated={terminated} truncated={truncated}"
        )

    env.close()
    if viewer:
        viewer.close()

    success_rate = successes / episodes
    success_rate_pct = 100.0 * success_rate

    print("\n===== TEST =====")
    print(f"Episodes:           {episodes}")
    print(f"Successes:          {successes}")
    print(f"Truncations (env):  {truncations}")
    print(f"Manual timeouts:    {manual_timeouts}")
    print(f"Success rate:       {success_rate_pct:.2f}%")
    print(f"Avg reward:         {np.mean(rewards_all):.2f}")
    print(f"Avg steps:          {np.mean(steps_all):.2f}")

    # plots
    plt.figure()
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards_all)
    plt.show()

    plt.figure()
    plt.title("Steps per episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.plot(steps_all)
    plt.show()



    return {
        "episodes": episodes,
        "successes": successes,
        "truncations": truncations,
        "manual_timeouts": manual_timeouts,
        "success_rate": success_rate,
        "success_rate_pct": success_rate_pct,
        "avg_reward": float(np.mean(rewards_all)),
        "avg_steps": float(np.mean(steps_all)),
    }



# ---------------------------
# Train/Test
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Taxi-v3 Q-learning (train/test) with fixed Matplotlib window")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train Q-table and save")
    p_train.add_argument("--episodes", type=int, default=5000)
    p_train.add_argument("--alpha", type=float, default=0.1)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--epsilon", type=float, default=0.1)
    p_train.add_argument("--save", type=str, default="q_table_taxi.npy")
    p_train.add_argument("--visualize", action="store_true", help="Show fixed window during training.")
    p_train.add_argument("--viz-every", type=int, default=200, help="View every N episodes")
    p_train.add_argument("--viz-max-steps", type=int, default=200)

    p_test = sub.add_parser("test", help="Load Q-table and play episodes")
    p_test.add_argument("--load", type=str, default="q_table_taxi.npy")
    p_test.add_argument("--episodes", type=int, default=15)
    p_test.add_argument("--visualize", action="store_true", help="Show fixed window during test.")
    p_test.add_argument("--no-viz", action="store_true", help="Run without a window (without Matplotlib)")
    p_test.add_argument("--max-steps", type=int, default=200)

    args = parser.parse_args()

    if args.mode == "train":
        q = train_q_learning(num_episodes=args.episodes,
                            alpha=args.alpha,
                            gamma=args.gamma,
                            epsilon=args.epsilon,
                            visualize=args.visualize,
                            viz_every=args.viz_every,
                            viz_max_steps=args.viz_max_steps,
                            )
        
        np.save(args.save, q)
        print(f"Salved: {args.save}  (shape={q.shape})")

    elif args.mode == "test":
        if not os.path.exists(args.load):
            raise FileNotFoundError(f"I couldn't find the file: {args.load}. First, run the train mode.")
        q = np.load(args.load)
        test_policy(q,
                    episodes=args.episodes,
                    # visualize=(not args.no_viz),
                    visualize=args.visualize,
                    max_steps=args.max_steps,
                    )

if __name__ == "__main__":
    main()
