import gym
from collections import Counter
import numpy as np
from statistics import median, mean
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim


# Generating training and test data from random actions
def generate_data(env_name, max_episodes, max_timesteps, score_requirment):
    '''
        This function generates training data from running the environment
        several times and taking random actions, If the random actions amount
        to a certain minimum score the the observation data from that environment
        along with its actions are stored.

        Args:
            env_name: str->Name of the environment
            max_episodes: int->should be very large > 5000
            max_time: int->Max steps the agent can take
            score_requirement: int->The minimum score a game should have to use it as training data

        Returns:
            final_data: shape->[[[observation], action], [[observation], action], .......]
                        dtype->np.ndarray
    '''

    raw_data = []  # [game_memory, game_memory, ........]
    accepted_score_list = []
    env = gym.make(env_name)
    print('Generating data.......\n')
    # Running multiple games
    for _ in tqdm(range(max_episodes)):
        score = 0  # tracking the score of the env
        prev_obs = []  # the action taken should be based on previous observation
        game_memory = []  # [[prev_obs], [action]] --> [[x, x, x, x], [left(0), right(1)]]

        # Running a single game
        env.reset()
        for _ in range(max_timesteps):
            # take action and collect observation and other stuff
            action = np.random.randint(2)
            observation, reward, done, info = env.step(action)

            # start saving previous observation, the first action run wont have a prev obs
            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action])

            prev_obs = observation  # update prev_observation
            score += reward  # add reward to score

            if done:
                break

        # Save data if the score is above min score requirement for the game
        if score > score_requirment:
            raw_data += game_memory[: -2]
            accepted_score_list.append(score)

    env.close()
    del prev_obs
    del observation
    del game_memory

    data = []
    for obs, action in raw_data:
        # if action == 0:
        #     action = [1,0]
        # elif action == 1:
        #     action = [0,1]
        data.append([obs.tolist(), action])
    del raw_data

    final_data = np.array(data, dtype='object')
    del data

    print(f"\nFound {len(final_data)} usable samples!")
    accepted_score_list.sort()
    print('Average accepted score:', mean(accepted_score_list))

    print('Median score for accepted scores:', median(accepted_score_list), end="\n")
    print(Counter(final_data[:, 1]))
    print(Counter(accepted_score_list))

    save = True if input('\nSave data?[y/n]: ') == 'y' else False
    if save:
        print('Saving data...')
        np.save(f'training_data_{env_name}.npy', final_data)
    else:
        print('data not saved!')

    print("done.\n")
    return final_data


class NN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input = nn.Linear(input_shape, 64)
        self.D1 = nn.Linear(64, 64)
        self.D2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.D1(x))
        x = F.relu(self.D2(x))
        return F.log_softmax(x, dim=1)


def train_model():
    data = generate_data(env_name, Episodes, time_steps, goal_reward)
    np.random.shuffle(data)  # shuffling data
    train = data[: int(pct * len(data))]  # splitting data into train based on pct
    train = train[: -(len(train) % BATCH_SIZE)]  # having the data disivible by the batch size
    test = data[int(pct * len(data)):]  # splitting data into test based on pct
    # converting data into torch tensors
    train_x, train_y = torch.Tensor([i[0] for i in train]).view(-1, 4), torch.Tensor([i[1] for i in train])
    test_x, test_y = torch.Tensor([i[0] for i in test]), torch.Tensor([i[1] for i in test])
    train_y = train_y.type(torch.long)  # need to change the labels to long for some reason, does not accept float

    print(f'Total samples: {len(data)}\nmod30 of 80%: {len(train)}')
    print(f'{len(train)} , {len(train_x)}, {len(train_y)}')
    input('\nPress ENTER to continue\n')

    model = NN(4)
    optimizer = optim.Adam(model.parameters())  # using adam optimizer
    loss_fn = F.nll_loss                        # using nll loss
    test_y = np.array([i.argmax().item() for i in test_y])

    for ep in range(EPOCHS):
        for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
            # making batch for each loop
            batch_X = train_x[i: i+BATCH_SIZE]
            batch_y = train_y[i: i+BATCH_SIZE]

            if torch.numel(batch_X) < 4:  # I forgot why I put this for
                print(f"breaking at: {i/30}")
                print(batch_X)
                break

            model.zero_grad()  # clear previous gradients
            outputs = model(batch_X)

            # back propogating the gradients
            loss = loss_fn(outputs, batch_y)
            # applying gradients
            loss.backward()
            optimizer.step()

        # validation
        preds = []
        for i in range(len(test_x)):
            preds.append(model.forward(test_x[i].view(-1, 4)).argmax().item())
        preds = np.array(preds)
        acc = round(sum(test_y == preds)/len(test_y), 4)  # round the validation accuracy to 4 decimal places
        print(f'Loss: {round(loss.item(), 4)}\tVal_acc: {acc}')

    # option to save the model
    save = True if input('\nSave model?[y/n]: ') == 'y' else False
    if save:
        torch.save(model.state_dict(), f'Models/CartPole-v0_model({acc}).pt')

    return model


def test_model(arg):
    '''
    arg can be a string containing the saved model path or
    it can be a model
    '''
    if type(arg) == str:
        model = NN(4)  # 4 is the observation length
        model.load_state_dict(torch.load(arg))
    else:
        model = arg

    env = gym.make(env_name)
    score_list = []

    for episode in range(10):
        observation = env.reset()
        score = 0
        for t in range(500):
            env.render()
            action = model.forward(torch.Tensor(observation.tolist()).view(-1, 4)).argmax().item()
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        print(f'Done in: {t}steps -> max is 499')
        score_list.append(score)
    env.close()
    print(f'Average Score: {sum(score_list)/len(score_list)}')


if __name__ == '__main__':
    env_name = 'CartPole-v0'

    # These parameters can be tweaked
    Episodes = 10000
    goal_reward = 50
    EPOCHS = 5
    BATCH_SIZE = 30

    # Don't tweak these
    time_steps = 500
    pct = 0.9

    # train a model and see how it performs
    model = train_model()
    test_model(model)

    # # test the models trained my me, PS. thest are graphical
    # test_model('Models/CartPole-v0_model(0.623).pt') # this is the best model I could train
    # test_model('Models/CartPole-v0_model(0.612).pt')
