from Argument.Parameter.Train_Parameters import Train_Parameters_args

args_Parameter = Train_Parameters_args()


def schedule_learning_rate(epoch, optimizer):

    new_lr = args_Parameter.learning_rate * ((1 - (epoch / args_Parameter.num_epochs)) ** 0.9)
    new_lr = max(new_lr, 1e-4)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr
    
# def decayed_learning_rate(step,
#                           initial_learning_rate,
#                           end_learning_rate,
#                           decay_steps,
#                           power=1.0):

#     step = min(step, decay_steps)
#     decayed = (1.0 - step / decay_steps) ** power
#     return (initial_learning_rate - end_learning_rate) * decayed + end_learning_rate