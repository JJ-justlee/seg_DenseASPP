from Argument.Parameter.Train_Parameters import Train_Parameters_args

args_Parameter = Train_Parameters_args()

def schedule_learning_rate(epoch, optimizer):

    new_lr = args_Parameter.learning_rate * ((1 - (epoch / args_Parameter.num_epochs)) ** 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr