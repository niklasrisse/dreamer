import math
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
import os

import phyre

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import imageio
import os as os
import io
import datetime
import uuid
import pathlib
import math
import argparse

def generate_data(args):

    eval_setup = 'ball_cross_template'

    fold_id =  0
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)

    action_tier = phyre.eval_setup_to_action_tier(eval_setup)

    tasks_2 = [x for x in train_tasks if x.startswith(args.template)==True]

    tasks = tasks_2
    simulator = phyre.initialize_simulator(tasks, action_tier)

    actions = simulator.build_discrete_action_space(max_actions=100000)

    try:
        os.mkdir('./data-generation/numpys')
    except:
        pass

    try:
        os.mkdir('./data-generation/episodes')
    except:
        pass

    for task_index in tqdm_notebook(range(1, len(tasks_2))):

        solution_found_counter = 0

        # 10 rollouts for each
        for k in range(10):

            if solution_found_counter < 5:

                action = random.choice(actions)
                simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True,
                                                       stride=4)

                num_trys = 0
                while (simulation.status != phyre.simulation_cache.SOLVED and num_trys < 10000):
                    num_trys += 1
                    action = random.choice(actions)
                    simulation = simulator.simulate_action(task_index, action, need_images=True,
                                                           need_featurized_objects=True, stride=4)
                    if str(simulation.status) == "SimulationStatus.INVALID_INPUT":
                        num_trys -= 1
            else:

                action = random.choice(actions)
                simulation = simulator.simulate_action(task_index, action, need_images=True, need_featurized_objects=True,
                                                       stride=4)

                while (simulation.status != phyre.simulation_cache.NOT_SOLVED):
                    action = random.choice(actions)
                    simulation = simulator.simulate_action(task_index, action, need_images=True,
                                                           need_featurized_objects=True, stride=4)

            print(simulation.status)
            filename = 'data-generation/numpys/task-00023:' + str(task_index + 1) + '_' + str(k) + '.npy'


            print(filename)
            np.save(filename, simulation.images)

            if str(simulation.status) == "SimulationStatus.SOLVED":
                solution_found_counter += 1

    dataset_path = './data-generation/numpys'

    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256

    max_episode_len = 120

    for filename in os.listdir(dataset_path):
        if filename.startswith('Task'):
            continue
        counter = 0
        print("Laded file: ", filename)
        data = os.path.join(dataset_path, filename)
        try:
            data = np.load(data)
        except:
            continue
        print(data.shape)

        if (len(data) >= max_episode_len):
            images = np.zeros((len(data), 64, 64, 3), dtype=np.uint8)
            rewards = np.ones((len(data)), dtype=np.float16)
            actions = np.ones((len(data), 6), dtype=np.float16)
            orientations = np.ones((len(data), 14), dtype=np.float16)
            velocity = np.ones((len(data), 9), dtype=np.float16)
            height = np.ones((len(data)), dtype=np.float16)
            discount = np.ones((len(data)), dtype=np.float16)


            for k, scene in enumerate(data):

                current_image = np.zeros((256, 256, 3), dtype=np.uint8)
                channel_0 = np.copy(np.flipud((scene)))
                channel_1 = np.copy(np.flipud((scene)))

                channel_0[channel_0 == 6] = 0
                channel_0[channel_0 > 0] = 255

                channel_1[channel_1 != 6] = 0
                channel_1[channel_1 == 6] = 255

                current_image[:, :, 0] = np.copy(channel_0)
                current_image[:, :, 1] = np.copy(channel_1)


                scaled_down_image = np.copy(current_image[::4,::4,:])
                images[k] = scaled_down_image

            numpy_dict = {"image": images[:max_episode_len], "action": actions[:max_episode_len], "reward": rewards[:max_episode_len], "orientations": orientations[:max_episode_len], "velocity": velocity[:max_episode_len], "dsicount": discount[:max_episode_len], "height": height[:max_episode_len]}

            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

            identifier = str(uuid.uuid4().hex)
            length = len(numpy_dict['reward'])
            directory = pathlib.Path('data-generation/episodes')
            filename = pathlib.Path(f'{timestamp}-{identifier}-{length}.npz')
            filename = directory / filename
            with io.BytesIO() as f1:
                np.savez_compressed(f1, **numpy_dict)
                f1.seek(0)
                with filename.open('wb') as f2:
                    f2.write(f1.read())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters.')
    parser.add_argument('--template', type=str, required=True)


    args = parser.parse_args()

    generate_data(args)
