from train_comm_game import main


writer = SummaryWriter()

parser = argparse.ArgumentParser(description="Trains the agents for cooperative communication task")
parser.add_argument('--n-epochs', '-e', type=int, help='if specified sets number of training epochs (default 5000)')
parser.add_argument('--learning-rate', type=float, help='if specified sets learning rate (default 1e-3)')
parser.add_argument('--batch-size', type=int, help='if specified sets batch size(default 256)')
parser.add_argument('--n-timesteps', '-t', type=int, help='if specified sets timestep length of each episode (default 32)')
parser.add_argument('--test-only', action='store_true', help='if specified only tests the specified model')
parser.add_argument('--ablation', type=str, default='default', help='if specified builds the agent as a specific ablation in [random]')
parser.add_argument('--train_set', type=str, default='all', help='[one, all]')

parser.add_argument('--load-model-weights', type=str, help='if specified start with saved model weights saved at file given by this argument')
parser.add_argument('--save-model-weights', type=str, help='if specified save the model weights at file given by this argument')
parser.add_argument('--use-cuda', action='store_true', help='if specified enables training on CUDA (default disabled)')

def print_losses(epoch, performance, image, set_name):
    loss = performance[set_name]['losses']
    rwd = performance[set_name]['rewards'] # sum of rewards at each timestep for one task.
    teacher_rwd = performance[set_name]['teacher_rwds']
    student_rwd = performance[set_name]['student_rwds']
    teacher_loss = performance[set_name]['teacher_losses']
    student_loss = performance[set_name]['student_losses']
    points = performance[set_name]['points']
    F1 = performance[set_name]['F1s']
    lr = performance['learning_rate'][-1]

    # Loss
    l, min_l = loss[-1], min(loss)
    t_l, min_t = teacher_loss[-1], min(teacher_loss)
    s_l, min_s = student_loss[-1], min(student_loss)

    # Reward
    r, max_r = rwd[-1], max(rwd)
    t_r, max_t = teacher_rwd[-1], max(teacher_rwd)
    s_r, max_s = student_rwd[-1], max(student_rwd)

    point = points[-1] if len(points) > 0 else 0
    max_point = max(points) if len(points) > 0 else 0

    F1_ = F1[-1] if len(F1) else -1
    max_F1 = max(F1)

    writer.add_scalar('Learning Rate', lr, epoch)
    writer.add_scalar('Reward/Total/{}'.format(set_name), r, epoch)
    writer.add_scalar('Reward/Teacher/{}'.format(set_name), t_r, epoch)
    writer.add_scalar('Reward/Student/{}'.format(set_name), s_r, epoch)
    writer.add_scalar('Loss/Total/{}'.format(set_name), l, epoch)
    writer.add_scalar('Loss/Teacher/{}'.format(set_name), t_l, epoch)
    writer.add_scalar('Loss/Student/{}'.format(set_name), s_l, epoch)
    writer.add_scalar('Points/{}'.format(set_name), point, epoch)
    writer.add_scalar('F1/{}'.format(set_name), F1_, epoch)

    if image is not None:
        writer.add_figure('Trajectories/{}'.format(set_name), image, epoch)

    print(set_name)
    print("[epoch %d][%d batches]" % (epoch, len(rwd)) +
          "[last point: %f][max point: %f][last F1: %f][max F1: %f]" % (point, max_point, F1_, max_F1) +
          "[last rwd: %f][max rwd: %f]" % (r, max_r) +
          "[last loss: %f][min loss: %f]" % (l, min_l) +
          "[last T rwd: %f][max T rwd: %f]" % (t_r, max_t) +
          "[last T loss: %f][min T loss: %f]" % (t_l, min_t) +
          "[last S rwd: %f][max S rwd: %f]" % (s_r, max_s) +
          "[last S loss: %f][min S loss: %f]" % (s_l, min_s)
          )
    print("_________________________")

"""
Returns dict w keys 'training', 'validation', 'ood_maps_validation', ood_tasks_validation and entries
that are dicts of {task_no: {task_str, task_features...}}
"""
def load_data():
    print('loading data')
    data = []
    for dataset in ['training', 'validation', 'ood_maps_validation', 'ood_tasks_validation']:
        with open(dataset + '.json') as f:
            data.append(json.load(f))
    return tuple(data)

def evaluate(agent, game, train=True):
    # print('train:66')
    if not train:
        agent.train(False)
        with torch.no_grad():
            attn_weights = agent(game)
            total_loss, teacher_loss, student_loss, total_rwd, teacher_rwd, student_rwd = get_loss(game)
            # total_loss, teacher_loss, student_loss, attn_weights = agent(game)
    elif train:
        agent.train()
        # print('train:73')
        attn_weights = agent(game)
        total_loss, teacher_loss, student_loss, total_rwd, teacher_rwd, student_rwd = get_loss(game)
        # total_loss, teacher_loss, student_loss, attn_weights = agent(game)

    return total_loss, teacher_loss, student_loss, total_rwd, teacher_rwd, student_rwd, attn_weights


def track_loss(epoch, point, F1, performance, total_loss, total_teacher_loss, total_student_loss,
               total_reward, total_teacher_reward, total_student_reward, image, set_name, write=False):
    per_agent_loss = total_loss.item()
    performance[set_name]['losses'].append(per_agent_loss)
    performance[set_name]['rewards'].append(total_reward.item())

    if total_teacher_loss is not None:
        total_teacher_loss = total_teacher_loss.item()
        performance[set_name]['teacher_losses'].append(total_teacher_loss)
        performance[set_name]['teacher_rwds'].append(total_teacher_reward.item())

    if total_student_loss is not None:
        total_student_loss = total_student_loss.item()
        performance[set_name]['student_losses'].append(total_student_loss)
        performance[set_name]['student_rwds'].append(total_student_reward.item())

    performance[set_name]['points'].append(point)
    performance[set_name]['F1s'].append(F1)

    if write: print_losses(epoch, performance, image, set_name)


def train(training_config, game_config, valid_game_config, agent, game, validation_games):
    print('starting training')

    optimizer = RMSprop(agent.parameters(), lr=training_config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.999, verbose=True, cooldown=5)

    # losses, points, F1 tracking
    performance = {}
    performance['learning_rate'] = []
    for set_name in ['train', 'validation', 'ood_tasks', 'ood_maps']:
        performance[set_name] = {}
        for to_log in ['losses', 'points', 'F1s', 'student_losses', 'teacher_losses', 'rewards', 'teacher_rwds', 'student_rwds']:
            performance[set_name][to_log] = []

    for epoch in range(training_config.num_epochs):
        if training_config.use_cuda:
            game.cuda()

        optimizer.zero_grad()
        # print('train:132')
        game.reset_memory(new_task=True) # reset each epoch to avoid hidden states spilling over to the next task
        agent.reset()
        # print('train:135')
        total_loss, total_teacher_loss, total_student_loss, \
                total_rwd, total_teacher_rwd, total_student_rwd, attn_weights = evaluate(agent, game)
        # print('train:137')
        # print('TOTAL LOSS', total_loss)

        total_loss /= game_config.batch_size
        total_rwd /= game_config.batch_size
        total_loss.backward()
        optimizer.step()

        # Tracking
        point = game.get_avg_score_for_last_task()
        F1 = game.get_avg_F1_for_last_task()
        if total_student_loss is not None:
            total_student_loss /= game_config.batch_size
            total_student_rwd /= game_config.batch_size
        if total_teacher_loss is not None:
            total_teacher_loss /= game_config.batch_size
            total_teacher_rwd /= game_config.batch_size

        # Do image every 100 epochs
        if training_config.log_image and epoch % 500 == 0:
            task_str = game.get_task_str()
            image = game.generate_images(attn_weights, task_str, epoch)
        else:
            image = None

        # Reduce LR when loss plateaus
        scheduler.step(total_loss.item())
        performance['learning_rate'].append(scheduler._last_lr[0])

        track_loss(epoch,
                   point,
                   F1,
                   performance,
                   total_loss,
                   total_teacher_loss,
                   total_student_loss,
                   total_rwd,
                   total_teacher_rwd,
                   total_student_rwd,
                   image,
                   'train', write=True)

        # Run on entire validation set in parallel
        if epoch % 500 == 0:
            test(validation_games, agent, valid_game_config, performance, write=True, epoch=epoch)


def test(validation_games, agent, game_config, performance, n_samples=1, write=False, epoch=None, report=False):
    # evaluate
    for n in tqdm(range(n_samples)):
        for validation in validation_games:
            valid_game = validation_games[validation]

            # run through all the tasks in the validation set.
            valid_game.reset_memory(new_task=True)
            agent.reset()
            total_loss, total_teacher_loss, total_student_loss, \
                total_rwd, total_teacher_rwd, total_student_rwd, attn_weights = evaluate(agent, valid_game, train=False)

            points = valid_game.get_avg_score_for_last_task()
            F1s = valid_game.get_avg_F1_for_last_task()

            avg_loss = total_loss / game_config.batch_size # TODO: this batch size needs to adjust to the len of tasks
            avg_rwd = total_rwd / game_config.batch_size
            avg_points = points / game_config.batch_size
            avg_F1 = F1s / game_config.batch_size

            if total_teacher_loss is not None:
                total_teacher_loss /= game_config.batch_size
                total_teacher_rwd /= game_config.batch_size
            if total_student_loss is not None:
                total_student_loss /= game_config.batch_size
                total_student_rwd /= game_config.batch_size

            task_str = valid_game.get_task_str()
            image = valid_game.generate_images(attn_weights, task_str, epoch)

            track_loss(
                n if epoch is None else epoch,
                avg_points,
                avg_F1,
                performance,
                avg_loss,
                total_teacher_loss,
                total_student_loss,
                avg_rwd,
                total_teacher_rwd,
                total_student_rwd,
                image,
                validation,
                write=write
            )

    if report:
        for set_name in performance:
            print('For set ', set_name)
            for metric in performance[set_name]:
                m = performance[set_name][metric]
                try:
                    print('Metric: ', metric)
                    print('Mean: {}, Std: {}, Min: {}, Max: {}'.format(np.mean(m), np.std(m), np.min(m), np.max(m)))
                except:
                    pass
            print('-----------------------------------------------------------------------------------------------')


def main():
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    valid_game_config = configs.get_valid_game_config(args)
    training_config = configs.get_training_config(args)
    print("Training with config:")
    print(training_config)
    print(game_config)
    print(valid_game_config)
    print(agent_config)

    # load data here
    training_set, validation_set, ood_maps_set, ood_tasks_set = load_data()

    if args['train_set'] == 'one':
        task = random.choice(list(training_set.items()))
        training_set = {task[0]: task[1]}
        # ood_maps_set = {task[0]: ood_maps_set[task[0]]}
        # validation_set = training_set

    if args['ablation'] == 'random':
        print('Using random agent')
        agent = RandomAgentModule(agent_config)
    elif args['ablation'] == 'default':
        print('Using default agent')
        # Both GRU modules with attention and teacher access FOL.
        agent = AgentModule(agent_config)
    elif args['ablation'] == 'default-with-direct-teacher-rwd':
        print('Using default agent with direct teacher loss in training')
    elif args['ablation'] == 'student-no-teacher':
        # TODO: one GRU module. in = visual features, out = actions
        print('Training student only')
        agent = AgentModule(agent_config)
    elif args['ablation'] == 'teacher-pretrain':
        # TODO: one GRU module. in = visual features, FOL, out = actions
        print('Training GRU module with FOL')
    elif args['ablation'] == 'student-random-teacher':
        # TODO: random actions for teacher and normal for student (with attention)
        print('Training student with random teacher')

    if training_config.use_cuda:
        agent.cuda()

    # Make games and send to cuda
    game = GameModule(game_config, training_set, random_select=True)

    if training_config.use_cuda:
        game.cuda()

    validation_games = {
        'validation': GameModule(valid_game_config, validation_set),
        'ood_tasks': GameModule(valid_game_config, ood_tasks_set),
        'ood_maps': GameModule(valid_game_config, ood_maps_set)
    }
    validation_games = {k: validation_games[k].cuda() if training_config.use_cuda else validation_games[k] for k in validation_games}

    if not args['test_only']:
        train(training_config, game_config, valid_game_config, agent, game, validation_games)
    else:
        performance = {}
        for set_name in ['train', 'validation', 'ood_tasks', 'ood_maps']:
            performance[set_name] = {}
            for to_log in ['losses', 'points', 'F1s', 'student_losses', 'teacher_losses', 'rewards', 'teacher_rwds', 'student_rwds']:
                performance[set_name][to_log] = []
        test(validation_games, agent, game_config, performance, n_samples=100, report=True)

    if training_config.save_model:
        torch.save(agent, training_config.save_model_file)
        print("Saved agent model weights at %s" % training_config.save_model_file)
    """
    import code
    code.interact(local=locals())
    """


if __name__ == "__main__":
    main()
    writer.close()

